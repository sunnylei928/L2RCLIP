import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from data.bases import ImageDataset

class LiverDataModule(pl.LightningDataModule):
    def __init__(self, train_images_root, train_data_file, val_images_root, val_data_file, test_images_root, test_data_file, batch_size=16, **kwargs):
        super().__init__()
        self.train_root = train_images_root
        self.train_file = train_data_file
        self.val_root = val_images_root
        self.val_file = val_data_file
        self.test_root = test_images_root
        self.test_file = test_data_file
        self.batch_size = batch_size

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
        data_list = []
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                full_path = os.path.join(img_root, parts[0])
                data_list.append((full_path, int(parts[1])))
        return data_list

    def setup(self, stage=None):
        self.train_set = ImageDataset(self.parse_txt_file(self.train_file, self.train_root), transform=self.train_transforms)
        self.val_set = ImageDataset(self.parse_txt_file(self.val_file, self.val_root), transform=self.val_transforms)
        self.test_set = ImageDataset(self.parse_txt_file(self.test_file, self.test_root), transform=self.val_transforms)

    def train_dataloader(self):
        # 极简模式：shuffle=True，开启 drop_last 防报错
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)