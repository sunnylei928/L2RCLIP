import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

# 导入你之前的基类和采样器
from .bases import ImageDataset

class LiverDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_images_root, 
        train_data_file, 
        val_images_root, 
        val_data_file, 
        test_images_root, 
        test_data_file, 
        batch_size=32, 
        num_instances=4, 
        **kwargs # 兼容 yaml 中其他的多余参数
    ):
        super().__init__()
        self.train_root = train_images_root
        self.train_file = train_data_file
        self.val_root = val_images_root
        self.val_file = val_data_file
        self.test_root = test_images_root
        self.test_file = test_data_file
        
        self.batch_size = batch_size
        self.num_instances = num_instances

        # ==========================================
        # 图像预处理与增强 (对齐 CLIP 的输入标准 224x224)
        # ==========================================
        self.train_transforms = T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop((224, 224)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.val_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def parse_txt_file(self, txt_path, img_root):
        """
        核心解析函数：逐行读取 txt 文件，组装绝对路径和标签
        """
        data_list = []
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 假设 txt 格式为 "image_name.jpg 2"
                parts = line.split()
                img_name = parts[0]
                label = int(parts[1])
                
                # 拼接完整的绝对路径
                full_path = os.path.join(img_root, img_name)
                data_list.append((full_path, label))
                
        return data_list

    def setup(self, stage=None):
        """
        Lightning 会在训练开始前自动调用此函数准备数据
        """
        train_data = self.parse_txt_file(self.train_file, self.train_root)
        val_data = self.parse_txt_file(self.val_file, self.val_root)
        test_data = self.parse_txt_file(self.test_file, self.test_root)

        # 封装进我们之前写的 ImageDataset 中
        self.train_set = ImageDataset(train_data, transform=self.train_transforms)
        self.val_set = ImageDataset(val_data, transform=self.val_transforms)
        self.test_set = ImageDataset(test_data, transform=self.val_transforms)

    def train_dataloader(self):
            # 【修改】去掉 RandomIdentitySampler，直接使用原生 DataLoader 和 shuffle=True
            return DataLoader(
                self.train_set, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=4, 
                pin_memory=True,
                drop_last=True # 丢弃最后凑不满 16 个的零头，防止维度算错
            )

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)