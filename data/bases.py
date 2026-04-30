from PIL import Image, ImageFile
from torch.utils.data import Dataset
import os.path as osp
import random
import torch

# 防止加载截断的图像时报错
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 【核心修改】针对超声肝纤维化分级的专用医学词汇表 (F0 - F4)
# 这里的文本将作为 CLIP 文本编码器的输入，与图像特征进行对齐
# ==========================================
vocab = {
    0: "A healthy liver ultrasound image without fibrosis (F0 stage).",
    1: "An ultrasound image showing mild liver fibrosis (F1 stage).",
    2: "An ultrasound image showing moderate liver fibrosis (F2 stage).",
    3: "An ultrasound image showing severe liver fibrosis (F3 stage).",
    4: "An ultrasound image showing liver cirrhosis (F4 stage)."
}

def read_image(img_path):
    """
    持续尝试读取图像直到成功。
    这可以避免在大量 IO 进程中引发的 IOError。
    """
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class BaseDataset(object):
    """
    数据集的基类，用于统计类别和数量
    """
    def get_imagedata_info(self, data):
        ranks = []
        for _, rank in data:
            ranks += [rank]
        ranks = set(ranks)
        num_ranks = len(ranks)
        num_imgs = len(data)
        return num_ranks, num_imgs

    def print_dataset_statistics(self):
        raise NotImplementedError

class BaseImageDataset(BaseDataset):
    """
    图像数据集的基类，提供打印数据集统计信息的标准方法
    """
    def print_dataset_statistics(self, train, test):
        num_train_ranks, num_train_imgs = self.get_imagedata_info(train)
        num_test_ranks, num_test_imgs = self.get_imagedata_info(test)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ranks | # images | ")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | ".format(num_train_ranks, num_train_imgs))
        print("  test     | {:5d} | {:8d} | ".format(num_test_ranks, num_test_imgs))
        print("  ----------------------------------------")

class ImageDataset(Dataset):
    """
    标准 PyTorch Dataset 封装，负责具体的数据读取和预处理
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.vocab = vocab  # 绑定上面定义的医学分级词汇表

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, rank = self.dataset[index]
        try:
            img = read_image(img_path)
        except Exception as e:
            # 如果图像损坏，随机抽取另一张图像以保证训练不断开
            index = random.randint(0, len(self.dataset) - 1)
            img = read_image(self.dataset[index][0])
            print(f"read strange image:{img_path}, replaced with another. Error: {e}")
            
        if self.transform is not None:
            img = self.transform(img)
            
        return img, rank