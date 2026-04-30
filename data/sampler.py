from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    随机采样 N 个类别（Identities），然后针对每个类别随机采样 K 个实例。
    因此最终的 Batch Size = N * K。
    
    参数:
    - data_source (list): 包含 (img_path, label) 的列表。
    - batch_size (int): 一个批次的样本总数。
    - num_instances (int): 每个类别在批次中出现的次数（即 K）。
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        # 计算一个 batch 中包含多少个不同的类别
        self.num_pids_per_batch = self.batch_size // self.num_instances
        
        # 建立索引字典：{类别标签: [样本索引1, 样本索引2, ...]}
        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # 估算一个 epoch 中的总样本量，确保能被 num_instances 整除
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        # 为每个类别准备好打乱后的样本索引组
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            # 如果样本数不足 K 个，则进行有放回抽样凑齐
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        # 循环抽取类别填充 batch
        while len(avai_pids) >= self.num_pids_per_batch:
            # 随机选出 N 个类别
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                # 每个类别弹出 K 个实例放入 final_idxs
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                # 如果该类别的样本被抽干了，则从可用类别列表中移除
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length