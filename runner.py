from multiprocessing.sharedctypes import Value
from pathlib import Path
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import sys
import pandas as pd
from typing import Dict, List, Any
import numpy as np
sys.path.insert(0, '/home/ubuntu/lq/OrdinalCLIP')

from ordinalclip.utils.logging import get_logger
from ordinalclip.runner.optim.lr_scheduler import build_lr_scheduler
from ordinalclip.runner.optim.optimizer import build_optimizer
from ordinalclip.runner.utils import load_pretrained_weights

import matplotlib.pyplot as plt
from losses.l2r_losses import L2RLosses
from models.ordinalclip import OrdinalCLIP

logger = get_logger(__name__)

class L2RCLIPRunner(pl.LightningModule):
    def __init__(
        self,
        model_cfg,
        output_dir: str,
        optimizer_and_scheduler_cfg,
        load_weights_cfg,
        seed: int,
        loss_weights=dict(ce_loss=1.0, cop_loss=1.0),
        stage2_start_epoch=10, 
        ckpt_path="",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # 这里哪怕加载了旧的 OrdinalCLIP 也没关系
        # 不修改，直接实例化内部 OrdinalCLIP，不通过外部 MODELS.build()
        # 之前 MODELS.build() 实际调用的是外部 OrdinalCLIP 仓库的代码，这里修改后全部生效
        model_cfg_dict = dict(model_cfg)
        model_cfg_dict.pop("type", None)
        self.module = OrdinalCLIP(**model_cfg_dict)
        self.loss_weights = loss_weights
        self.num_ranks = getattr(self.module, "num_ranks", 5) 

        logger.info("内部 OrdinalCLIP 加载成功，使用修改后的 dropout 和 TextEncoder")

        self.register_buffer("rank_output_value_array", torch.arange(0, self.num_ranks).float(), persistent=False)
        self.output_dir = Path(output_dir)
        self._custom_logger = get_logger(__name__)

        self.test_predictions = []
        self.load_weights(**load_weights_cfg)
        self._optimizer_and_scheduler_cfg = optimizer_and_scheduler_cfg
        self.seed = seed

    def forward(self, images):
        return self.module(images)

    def on_train_epoch_start(self) -> None:
        """
        严格复现论文的两阶段训练策略:
        Stage 1: 冻结 Image Encoder，训练 PromptLearner 和 RankFormer
        Stage 2: 解冻 Image Encoder，进行全局序数对齐微调
        """
        stage2_start = self.hparams.stage2_start_epoch

        if not hasattr(self.module, "image_encoder"):
            return

        image_encoder = self.module.image_encoder

        if self.current_epoch < stage2_start:
            # Stage 1: 冻结视觉编码器
            if image_encoder.training:  # 仅在首次进入时设置，避免重复操作
                image_encoder.requires_grad_(False)
                image_encoder.eval()  # 同时设为 eval 模式，防止 BN/Dropout 更新
                if self.current_epoch == 0:
                    logger.info(f"[Stage 1] Epoch {self.current_epoch}: 视觉编码器已锁定 (requires_grad=False)")
        else:
            # Stage 2: 解冻视觉编码器
            if not image_encoder.training:  # 仅在切换时设置
                image_encoder.requires_grad_(True)
                image_encoder.train()  # 恢复训练模式
                logger.info(f"[Stage 2] Epoch {self.current_epoch}: 视觉编码器已解冻 (requires_grad=True)")

                # 关键: 优化器在一开始就已包含 image_encoder 参数（见 build_param_dict），
                # 因此解冻后无需重新初始化优化器，Adam 的动量缓冲区和学习率调度器状态得以保留。

    def run_step(self, batch, batch_idx, step_type: str = "train"):
        """
        通用步骤执行函数，兼容 train/val/test 阶段
        :param batch: 数据批次
        :param batch_idx: 批次索引
        :param step_type: 阶段类型 (train/val/test)
        """
        x, y = batch
        # 调用 OrdinalCLIP，获取三元组输出
        logits, img_feats, txt_feats = self.module(x)

        # 核心损失计算（测试阶段不计算损失）
        losses = {}
        if step_type != "test":
            losses = self.compute_losses(logits, img_feats, txt_feats, y)
            total_loss = sum([self.loss_weights.get(k, 0.0) * losses[k] for k in losses])
        else:
            total_loss = torch.tensor(0.0, device=self.device)

        # 计算评估指标 (MAE)
        metrics_exp = self.compute_per_example_metrics(logits, y, "exp")
        metrics_argmax = self.compute_per_example_metrics(logits, y, "argmax")

        # 测试阶段收集预测结果（含F0-F4概率）
        if step_type == "test":
            self.collect_test_predictions(logits, y)

        return {
            "loss": total_loss, 
            **losses, 
            **metrics_exp,
            **metrics_argmax
        }

    def compute_losses(self, logits, img_feats, txt_feats, y):
        losses = {}

        # 1. Asymmetrical Contrastive Loss
        # 对应论文：允许一图对多文，增强语义鲁棒性
        if self.loss_weights.get("ce_loss", 0) > 0:
            losses["ce_loss"] = L2RLosses.asymmetrical_contrastive_loss(
                txt_feats, img_feats, y, self.module.logit_scale
            )

        # 2. Cross-modal Ordinal Pairwise Loss (L_cop)
        # 对应论文：强制图像特征与文本等级锚点的距离随等级差单调增加
        if self.loss_weights.get("cop_loss", 0) > 0:
            losses["cop_loss"] = L2RLosses.ordinal_pairwise_loss(
                logits, img_feats, txt_feats, y, self.num_ranks
            )
        # 3. 新增：文本特征分散损失，防止5个等级的文本特征坍缩到同一个点
        # 当文本特征平均余弦相似度 > 0.5 时触发损失，使其互相远离
        if self.loss_weights.get("dispersion_loss", 0) > 0:
            txt_feats_norm = F.normalize(txt_feats, dim=-1)
            sim_matrix = txt_feats_norm @ txt_feats_norm.t()  # (5, 5)
            mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
            avg_sim = sim_matrix[mask].mean()
            losses["dispersion_loss"] = torch.relu(avg_sim - 0.5)

        return losses

    def collect_test_predictions(self, logits, y):
        """
        收集测试阶段的完整预测结果（含F0-F4概率）
        :param logits: 模型输出的原始logits
        :param y: 真实标签
        """
        # 1. 保持 Tensor 计算阶段（速度最快，全在同一设备上运算）
        probs_tensor = F.softmax(logits, dim=-1)  # 保持 Tensor
        rank_values = self.rank_output_value_array.type(logits.dtype)

        pred_exp_tensor = torch.sum(probs_tensor * rank_values, dim=-1)
        pred_max_tensor = torch.argmax(logits, dim=-1)

        # 2. 统一转换到 Numpy 阶段（准备提取纯数字）
        probs_np = probs_tensor.detach().cpu().numpy()
        pred_exp_np = pred_exp_tensor.detach().cpu().numpy()
        pred_max_np = pred_max_tensor.detach().cpu().numpy()
        true_label_np = y.detach().cpu().numpy()

        # 3. 逐样本收集数据
        for idx in range(len(true_label_np)):
            pred_dict = {
                "true_label": true_label_np[idx],
                "pred_exp": pred_exp_np[idx],
                "pred_max": pred_max_np[idx],
                # 此时提取出来的都是纯粹的 python float 浮点数，写入 CSV 绝对安全
                "prob_F0": probs_np[idx][0],
                "prob_F1": probs_np[idx][1],
                "prob_F2": probs_np[idx][2],
                "prob_F3": probs_np[idx][3],
                "prob_F4": probs_np[idx][4]
            }
            self.test_predictions.append(pred_dict)

    def compute_per_example_metrics(self, logits, y, gather_type="exp"):
        """
        计算单样本指标（MAE），支持两种预测方式：
        - exp: 基于概率期望的连续值预测
        - argmax: 基于 argmax 的离散值预测
        """
        probs = F.softmax(logits, -1)
        rank_values = self.rank_output_value_array.type(logits.dtype)

        # 使用期待值计算连续等级
        if gather_type == "exp":
            predict_y = torch.sum(probs * rank_values, dim=-1)
        else:  # argmax
            predict_y = torch.argmax(probs, dim=-1).float()

        # 确保维度匹配（处理批量维度）
        y_float = y.float().to(predict_y.device)
        mae = torch.abs(predict_y - y_float).mean()
        return {f"mae_{gather_type}_metric": mae}

    def training_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx, step_type="train")
        # 对齐 run.py 中 plot_metrics 的日志名称（train/total_loss）
        self.log("train/total_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True)
        # 记录各分项损失和指标
        for k, v in outputs.items():
            if "loss" in k and k != "loss":
                self.log(f"train_{k}", v, on_step=True, on_epoch=True, prog_bar=True)
            if "metric" in k:
                self.log(f"train_{k}", v, on_step=True, on_epoch=True, prog_bar=True)
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx, step_type="val")
        # 对齐 run.py 中 ModelCheckpoint/EarlyStopping 的监控指标（val_loss）
        self.log("val_loss", outputs["loss"], on_step=False, on_epoch=True, prog_bar=True)
        # 记录各分项损失和指标
        for k, v in outputs.items():
            if "loss" in k and k != "loss":
                self.log(f"val_{k}", v, on_step=False, on_epoch=True, prog_bar=True)
            if "metric" in k:
                self.log(f"val_{k}", v, on_step=False, on_epoch=True, prog_bar=True)

        # 新增：监控文本特征区分度，计算5个等级文本特征的平均余弦相似度
        # 如果文本特征坍缩，此值会接近1.0；正常应在 0.3-0.8 之间
        with torch.no_grad():
            _, _, txt_feats = self.module(batch[0][:1])  # 取1张图触发forward获取txt_feats
            txt_feats = txt_feats / (txt_feats.norm(dim=-1, keepdim=True) + 1e-6)
            sim_matrix = txt_feats @ txt_feats.t()  # (5, 5)
            # 排除对角线，计算上三角平均
            mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
            avg_sim = sim_matrix[mask].mean()
            self.log("val_text_feature_avg_cosine", avg_sim, on_step=False, on_epoch=True, prog_bar=False)

        return outputs

    def test_step(self, batch, batch_idx):
        """测试阶段步骤：仅计算指标，不计算损失，收集全量预测信息"""
        outputs = self.run_step(batch, batch_idx, step_type="test")
        # 记录测试指标
        for k, v in outputs.items():
            if "metric" in k:
                self.log(f"test_{k}", v, on_step=False, on_epoch=True, prog_bar=True)
        return outputs

    def on_test_epoch_end(self):
        """测试 epoch 结束后：保存包含F0-F4概率的预测结果到 CSV 文件"""
        if not self.test_predictions:
            logger.warning("测试阶段未收集到任何预测结果")
            return

        # 保存预测结果到 output_dir
        pred_df = pd.DataFrame(self.test_predictions)
        save_path = self.output_dir / "test_predictions.csv"
        pred_df.to_csv(save_path, index=False)
        logger.info(f"测试预测结果（含F0-F4概率）已保存至: {save_path.absolute()}")

        # 重置预测结果列表（避免多轮测试重复）
        self.test_predictions = []

    def configure_optimizers(self):
        # 注入 RankFormer 参数到优化器
        # 字典用键索引替代点语法
        param_dict_ls = self.build_param_dict(**self._optimizer_and_scheduler_cfg["param_dict_cfg"])
        optimizer = build_optimizer(model=param_dict_ls, **self._optimizer_and_scheduler_cfg["optimizer_cfg"])
        scheduler = build_lr_scheduler(optimizer=optimizer, **self._optimizer_and_scheduler_cfg["lr_scheduler_cfg"])
        return [optimizer], [scheduler]

    def build_param_dict(self, **kwargs):
        param_dict_ls = []

        # 1. 提示词上下文参数
        if kwargs.get("lr_prompt_learner_context", 0) > 0 and hasattr(self.module.prompt_learner, "context_embeds"):
            param_dict_ls.append({
                "params": [self.module.prompt_learner.context_embeds],  # 注意加上 []
                "lr": kwargs["lr_prompt_learner_context"], 
                "name": "prompt_context"
            })

        # 2. 基础等级嵌入参数
        if kwargs.get("lr_prompt_learner_ranks", 0) > 0 and hasattr(self.module.prompt_learner, "rank_embeds"):
            param_dict_ls.append({
                "params": [self.module.prompt_learner.rank_embeds],  # 注意加上 []
                "lr": kwargs["lr_prompt_learner_ranks"], 
                "name": "prompt_ranks"
            })

        # 3. 核心：RankFormer 参数注入
        if hasattr(self.module.prompt_learner, 'rankformer'):
            lr_rf = kwargs.get("lr_rankformer", kwargs.get("lr_prompt_learner_ranks", 0.0001))
            param_dict_ls.append({
                "params": self.module.prompt_learner.rankformer.parameters(), 
                "lr": lr_rf, 
                "name": "rankformer"
            })

        # 4. 视觉编码器参数 (Stage 2 使用，增加属性存在性检查)
        # 关键修正：无论 Stage 1 是否需要训练，都先将视觉编码器参数加入优化器。
        # 两阶段冻结仅通过 requires_grad_ 控制，避免 Stage 2 中途重新初始化优化器
        # 导致学习率状态、动量缓冲区丢失（尤其影响 Adam 类优化器）。
        if hasattr(self.module, "image_encoder"):
            lr_ie = kwargs.get("lr_image_encoder", 0.0)
            # 若未配置 image_encoder 学习率，继承 prompt_learner_context 的学习率作为保底
            if lr_ie <= 0:
                lr_ie = kwargs.get("lr_prompt_learner_context", 1e-4)

            param_dict_ls.append({
                "params": self.module.image_encoder.parameters(), 
                "lr": lr_ie, 
                "name": "image_encoder"
            })

        return param_dict_ls

    def load_weights(self, **kwargs):
        if kwargs.get("init_image_encoder_weights"):
            load_pretrained_weights(
                self.module.image_encoder, 
                kwargs["init_image_encoder_weights"]  # 删掉 strict=False
            )