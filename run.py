import argparse

import sys
from pathlib import Path

import torch
# 👇 【新增：性能优化】开启 NVIDIA L40 显卡 Tensor Cores 的狂暴加速模式 
torch.set_float32_matmul_precision('medium')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import confusion_matrix, mean_absolute_error
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

# ==========================================
# 导入我们自定义的 L2RCLIP 模块
# ==========================================
from data.datamodule import LiverDataModule
from runner import L2RCLIPRunner
import logging

logger = logging.getLogger(__name__)


def setup_logger(output_dir):
    """配置标准的 Python 文本日志，同时输出到屏幕和文件"""
    # 1. 定义和你原来一模一样的日志格式
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 2. 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 为了防止多次调用重复打印，先清空现有的 handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 3. 添加控制台输出 (StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    # 4. 添加文件输出 (FileHandler)
    if output_dir:
        # 在输出目录下生成一个纯文本日志文件
        log_file = output_dir / "training_process.log"
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)



def main(cfg: DictConfig):
    # ==========================================
    # 1. 初始化与日志环境设置
    # ==========================================
    pl.seed_everything(cfg.runner_cfg.seed, workers=True)

    output_dir = Path(cfg.runner_cfg.output_dir)

    setup_logger(output_dir)
    logger = logging.getLogger(__name__)
    logger.info("📝 纯文本运行日志系统已启动！")
    
    callbacks = load_callbacks(output_dir)
    loggers = load_loggers(output_dir)

    deterministic = True
    logger.info(f"`deterministic` flag: {deterministic}")

    # ==========================================
    # 2. 实例化 Trainer
    # ==========================================
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        deterministic=deterministic,
        **OmegaConf.to_container(cfg.trainer_cfg), 
    )

    # ==========================================
    # 3. 初始化数据模块与模型 Runner
    # ==========================================
    data_module = LiverDataModule(**OmegaConf.to_container(cfg.data_cfg))
    runner = None

    # ==========================================
    # 4. 训练流程 (Train)
    # ==========================================
    if not cfg.test_only:
        runner = L2RCLIPRunner(**OmegaConf.to_container(cfg.runner_cfg))

        logger.info("🚀 Start training L2RCLIP for Liver Fibrosis Grading...")
        trainer.fit(model=runner, datamodule=data_module)
        logger.info("🏁 End training.")

    # ==========================================
    # 5. 测试流程 (Test)
    # ==========================================
    output_dir_ckpt_paths = output_dir / "ckpts"
    ckpt_paths = list(output_dir_ckpt_paths.glob("*.ckpt"))

    if len(ckpt_paths) == 0:
        logger.info("Zero-shot testing (No checkpoints found).")
        if runner is None:
            runner = L2RCLIPRunner(**OmegaConf.to_container(cfg.runner_cfg))
        trainer.test(model=runner, datamodule=data_module)
        logger.info("End zero-shot testing.")

    for ckpt_path in ckpt_paths:
        logger.info(f"Start testing ckpt: {ckpt_path}")
        runner = L2RCLIPRunner.load_from_checkpoint(str(ckpt_path), **OmegaConf.to_container(cfg.runner_cfg))
        trainer.test(model=runner, datamodule=data_module)
        logger.info(f"End testing ckpt: {ckpt_path}")

    # ==========================================
    # 6. 生成分析报告与图像
    # ==========================================
    plot_metrics(output_dir)
    generate_analysis_report(output_dir)


def plot_metrics(output_dir):
    output_path = Path(output_dir)
    csv_paths = list(output_path.rglob("metrics.csv"))
    
    if not csv_paths:
        logger.warning(f"在 {output_dir} 中未找到指标文件")
        return

    csv_logger_path = csv_paths[-1]
    metrics = pd.read_csv(csv_logger_path)

    train_loss = metrics.get("train/total_loss_epoch", pd.Series(dtype=float)).dropna().reset_index(drop=True)
    val_loss = metrics.get("val_loss", pd.Series(dtype=float)).dropna().reset_index(drop=True)

    train_mae_argmax = metrics.get("train_mae_argmax_metric_epoch", pd.Series(dtype=float)).dropna().reset_index(drop=True)
    val_mae_argmax = metrics.get("val_mae_argmax_metric", pd.Series(dtype=float)).dropna().reset_index(drop=True)

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    if not train_loss.empty:
        plt.plot(train_loss, label="Train Total Loss", color="#1f77b4", marker='o', markersize=4)
    if not val_loss.empty:
        plt.plot(val_loss, label="Validation Loss", color="#ff7f0e", marker='s', markersize=4)
    plt.ylabel("Loss Value")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(2, 1, 2)
    if not train_mae_argmax.empty:
        plt.plot(train_mae_argmax, label="Train MAE (Argmax)", color="#2ca02c", marker='o', markersize=4)
    if not val_mae_argmax.empty:
        plt.plot(val_mae_argmax, label="Val MAE (Argmax)", color="#d62728", marker='s', markersize=4)
    
    plt.xlabel("Epoch")
    plt.ylabel("MAE (Lower is Better)")
    plt.title("Classification Error (MAE Argmax) Over Epochs")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = output_path / "training_metrics.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"📊 指标综合曲线图已保存至: {save_path.absolute()}")


def generate_analysis_report(output_dir):
    pred_path = Path(output_dir) / "test_predictions.csv"
    
    if not pred_path.exists():
        logger.warning(f"未找到预测数据文件 {pred_path}，请确保 Runner 的 test_step 导出了结果。")
        return

    df = pd.read_csv(pred_path)
    y_true = df["true_label"]
    y_pred = df["pred_max"] 
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['F0', 'F1', 'F2', 'F3', 'F4'],
                yticklabels=['F0', 'F1', 'F2', 'F3', 'F4'])
    plt.xlabel('Predicted (Model)')
    plt.ylabel('True (Ground Truth)')
    plt.title('Liver Fibrosis Grading (F0-F4) - Confusion Matrix')
    
    cm_save_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(cm_save_path, dpi=150)
    plt.close()
    
    if "pred_exp" in df.columns:
        mae_val = mean_absolute_error(y_true, df["pred_exp"])
        logger.info(f"\n" + "*"*20 + " 深度分析报告 " + "*"*20)
        logger.info(f">>> 混淆矩阵图像已保存至: {cm_save_path}")
        logger.info(f">>> 测试集最终 MAE (Expected Value): {mae_val:.4f}")
        logger.info("*"*54 + "\n")


def load_loggers(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "csv_logger").mkdir(exist_ok=True, parents=True)
    (output_dir / "tb_logger").mkdir(exist_ok=True, parents=True) # 为 TensorBoard 创建文件夹
    
    loggers = []

    # 1. 挂载 CSV Logger
    loggers.append(
        pl_loggers.CSVLogger(
            str(output_dir),
            name="csv_logger",
            version=""
        )
    )
    
    # 👇 【新增：挂载 TensorBoard Logger】
    # 2. 挂载 TensorBoard Logger
    loggers.append(
        pl_loggers.TensorBoardLogger(
            str(output_dir),
            name="tb_logger",
            version=""
        )
    )

    return loggers

def load_callbacks(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "ckpts").mkdir(exist_ok=True, parents=True)

    callbacks = []
    
    callbacks.append(
        ModelCheckpoint(
            monitor="val_loss", 
            dirpath=str(output_dir / "ckpts"),
            filename="l2rclip-{epoch:02d}-{val_loss:.4f}", 
            verbose=True,
            save_last=False,
            save_top_k=1,
            mode="min",
            save_weights_only=False,
        )
    )

    callbacks.append(
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=20,
            verbose=True,
            mode="min"
        )
    )
    
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    return callbacks


def setup_output_dir_for_training(output_dir):
    output_dir = Path(output_dir)
    if output_dir.stem.startswith("version_"):
        output_dir = output_dir.parent
    output_dir = output_dir / f"version_{get_version(output_dir)}"
    return output_dir


def get_version(path: Path):
    versions = list(path.glob("version_*"))
    if not versions:
        return 0
    existing_versions = []
    for v in versions:
        try:
            existing_versions.append(int(v.name.replace("version_", "")))
        except ValueError:
            continue
    return max(existing_versions) + 1 if existing_versions else 0


def parse_cfg(args, instantialize_output_dir=True):
    cfg = OmegaConf.merge(*[OmegaConf.load(config_) for config_ in args.config])
    extra_cfg = OmegaConf.from_dotlist(args.cfg_options)
    cfg = OmegaConf.merge(cfg, extra_cfg)

    output_dir = Path(cfg.runner_cfg.output_dir if args.output_dir is None else args.output_dir)
    if instantialize_output_dir:
        if not args.test_only:
            output_dir = setup_output_dir_for_training(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    seed = args.seed if args.seed is not None else cfg.runner_cfg.seed
    cli_cfg = OmegaConf.create(
        dict(
            config=args.config,
            test_only=args.test_only,
            runner_cfg=dict(seed=seed, output_dir=str(output_dir)),
            trainer_cfg=dict(fast_dev_run=args.debug),
        )
    )
    cfg = OmegaConf.merge(cfg, cli_cfg)
    
    if instantialize_output_dir:
        OmegaConf.save(cfg, str(output_dir / "config.yaml"))
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", action="append", type=str, default=[])
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--cfg_options",
        default=[],
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    
    if not args.config:
        print("⚠️ 警告: 未提供配置文件。请使用 --config 路径指定 yaml 文件，例如：")
        print("python main.py --config configs/liver_l2rclip.yaml")
        sys.exit(1)

    cfg = parse_cfg(args, instantialize_output_dir=True)

    logger.info("====== Start L2RCLIP Pipeline ======")
    main(cfg)
    logger.info("====== End L2RCLIP Pipeline ======")