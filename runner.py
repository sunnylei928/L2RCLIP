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

from ordinalclip.models import MODELS

from ordinalclip.utils.logging import get_logger
from ordinalclip.runner.optim.lr_scheduler import build_lr_scheduler
from ordinalclip.runner.optim.optimizer import build_optimizer
from ordinalclip.runner.utils import freeze_param, load_pretrained_weights

import matplotlib.pyplot as plt
from losses.l2r_losses import L2RLosses

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
        self.module = MODELS.build(model_cfg)
        self.loss_weights = loss_weights
        self.num_ranks = getattr(self.module, "num_ranks", 5) 

        # =====================================================================
        # 🚀 终极自愈补丁：动态强行注入 L2RPromptLearner (无视路径冲突)
        # =====================================================================
        if not hasattr(self.module.prompt_learner, "rankformer"):
            logger.warning("⚠️ 检测到模型加载了旧版 PromptLearner，正在启动动态替换...")
            
            # 1. 确保能从你当前的新项目里导入 L2RPromptLearner
            import sys
            if '/home/ubuntu/lq/L2RCLIP' not in sys.path:
                sys.path.append('/home/ubuntu/lq/L2RCLIP')
            from models.prompt_learner import L2RPromptLearner
            
            # 2. 抓取底层的 CLIP 实例和特征维度
            internal_clip = getattr(self.module, "clip_model", getattr(self.module, "model", None))
            
            # 👇 【关键新增：防丢机制】如果旧模型把它丢了，我们当场加载一个用来提取词向量
            if internal_clip is None:
                import clip
                # 尝试从配置获取 backbone 名字，默认兜底使用 ViT-B/16
                backbone_name = model_cfg.get("text_encoder_name", "ViT-B/16")
                logger.info(f"🔄 旧版代码未保存 CLIP 实例，正从本地缓存加载 {backbone_name} 提取医学词嵌入...")
                internal_clip, _ = clip.load(backbone_name, device="cpu")
                internal_clip.float() # 保持数据类型一致
            # 👆 =======================================================================
            
            embed_dims = getattr(self.module, "embed_dims", 512)
            
            # 3. 强行覆盖为你的医学医学先验学习器
            self.module.prompt_learner = L2RPromptLearner(
                clip_model=internal_clip, 
                num_ranks=self.num_ranks, 
                embeddings_dim=embed_dims
            )
            
            # 4. 同步 Token 供外部 TextEncoder 提取句尾 [EOT]
            self.module.psudo_sentence_tokens = self.module.prompt_learner.psudo_sentence_tokens
            
            # 5. 动态截断 TextEncoder 的位置编码 (解决 17 vs 77 的 RuntimeError)
            seq_len = self.module.psudo_sentence_tokens.shape[1]
            orig_pos = self.module.text_encoder.positional_embedding
            if orig_pos.shape[0] != seq_len:
                logger.info(f"✂️ 正在动态截断位置编码 ({orig_pos.shape[0]} -> {seq_len}) 以完美匹配医学 Prompt...")
                self.module.text_encoder.positional_embedding = torch.nn.Parameter(
                    orig_pos[:seq_len, :].clone()
                )
#                👇 【关键新增：解决 77x77 报错】同步截断所有 Transformer 层的 Attention Mask
                if hasattr(self.module.text_encoder, "transformer"):
                    for block in self.module.text_encoder.transformer.resblocks:
                        if hasattr(block, "attn_mask") and block.attn_mask is not None:
                            block.attn_mask = block.attn_mask[:seq_len, :seq_len]
                # 👆 ===============================================================
                
            logger.info("✅ 动态替换成功！RankFormer 和医学先验 (F0-F4) 已全副武装上线。")
        else:
            logger.info("✅ 原生装配成功！已检测到 RankFormer。")
        # =====================================================================
        
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
        Stage 1: 冻结 Image Encoder，训练 PromptLearner 和 RankFormer。
        Stage 2: 解冻 Image Encoder，进行全局序数对齐微调。
        """
        # 获取配置中的开启时间
        stage2_start = self.hparams.stage2_start_epoch
        
        if hasattr(self.module, "image_encoder"):
            if self.current_epoch < stage2_start:
                # Stage 1: 保持视觉特征稳定
                self.module.image_encoder.requires_grad_(False)
                if self.current_epoch == 0:
                    logger.info(f"🛡️ [Stage 1] Epoch {self.current_epoch}: 视觉编码器已锁定。")
            else:
                # Stage 2: 开启跨模态微调
                self.module.image_encoder.requires_grad_(True)
                if self.current_epoch == stage2_start:
                    logger.info(f"🔥 [Stage 2] Epoch {self.current_epoch}: 视觉编码器已解冻。")

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

        return losses

    def collect_test_predictions(self, logits, y):
        """
        收集测试阶段的完整预测结果（含F0-F4概率）
        :param logits: 模型输出的原始logits
        :param y: 真实标签
        """
        # 1. 纯 Tensor 计算阶段（速度最快，全在同一设备上运算）
        probs_tensor = F.softmax(logits, dim=-1)  # 保持为 Tensor
        rank_values = self.rank_output_value_array.type(logits.dtype)
        
        pred_exp_tensor = torch.sum(probs_tensor * rank_values, dim=-1)
        pred_max_tensor = torch.argmax(logits, dim=-1)
        
        # 2. 统一转换为 Numpy 阶段（准备提取纯数字）
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
            logger.warning("测试阶段未收集到任何预测结果！")
            return
        
        # 保存预测结果到 output_dir
        pred_df = pd.DataFrame(self.test_predictions)
        save_path = self.output_dir / "test_predictions.csv"
        pred_df.to_csv(save_path, index=False)
        logger.info(f"📝 测试预测结果（含F0-F4概率）已保存至: {save_path.absolute()}")
        
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
                "params": [self.module.prompt_learner.context_embeds], # ✅ 加上 []
                "lr": kwargs["lr_prompt_learner_context"], 
                "name": "prompt_context"
            })
        
        # 2. 基础等级嵌入参数
        if kwargs.get("lr_prompt_learner_ranks", 0) > 0 and hasattr(self.module.prompt_learner, "rank_embeds"):
            param_dict_ls.append({
                "params": [self.module.prompt_learner.rank_embeds], # ✅ 加上 []
                "lr": kwargs["lr_prompt_learner_ranks"], 
                "name": "prompt_ranks"
            })

        # 3. 【核心】RankFormer 参数注入
        if hasattr(self.module.prompt_learner, 'rankformer'):
            lr_rf = kwargs.get("lr_rankformer", kwargs.get("lr_prompt_learner_ranks", 0.0001))
            param_dict_ls.append({
                "params": self.module.prompt_learner.rankformer.parameters(), 
                "lr": lr_rf, 
                "name": "rankformer"
            })

        # 4. 视觉编码器参数 (Stage 2 使用，增加属性存在性检查)
        if kwargs.get("lr_image_encoder", 0) > 0 and hasattr(self.module, "image_encoder"):
            param_dict_ls.append({
                "params": self.module.image_encoder.parameters(), 
                "lr": kwargs["lr_image_encoder"], 
                "name": "image_encoder"
            })
        else:
            if hasattr(self.module, "image_encoder"):
                freeze_param(self.module.image_encoder)
            
        return param_dict_ls

    def load_weights(self, **kwargs):
        if kwargs.get("init_image_encoder_weights"):
            load_pretrained_weights(
                self.module.image_encoder, 
                kwargs["init_image_encoder_weights"]  # 删掉 strict=False
            )