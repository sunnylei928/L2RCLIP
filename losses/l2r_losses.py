import torch
import torch.nn as nn
import torch.nn.functional as F

class L2RLosses:
    @staticmethod
    def asymmetrical_contrastive_loss(txt_feats, img_feats, labels, logit_scale):
        # 对应 supcontrast.py 的逻辑
        batch_size = img_feats.shape[0]
        t_label = torch.arange(txt_feats.shape[0]).to(img_feats.device)
        
        # 【修复维度报错】：labels.unsqueeze(1) -> (32, 1) | t_label.unsqueeze(0) -> (1, 5)
        # 生成的 mask 形状为 (batch_size, num_ranks)，例如 (32, 5)
        mask = torch.eq(labels.unsqueeze(1), t_label.unsqueeze(0)).float()
        
        # 计算预测相似度 logits (32, 5)
        logits = (img_feats @ txt_feats.t()) * logit_scale.exp()
        log_prob = F.log_softmax(logits, dim=1) # (32, 5)
        
        # 计算正样本的平均对数似然
        # 此时 mask (32, 5) 和 log_prob (32, 5) 可以完美相乘了
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        return -mean_log_prob_pos.mean()

    @staticmethod
    def ordinal_pairwise_loss(logits, img_feats, txt_feats, labels, num_ranks):
        # 对应 ordinal_ce.py 的 L_cop 逻辑
        age_list = torch.arange(num_ranks).float().to(logits.device)
        
        # 计算物理距离权重：labels.unsqueeze(1) 是 (32, 1), age_list.unsqueeze(0) 是 (1, 5)
        # dist_weight 形状为 (32, 5)
        dist_weight = torch.abs(labels.unsqueeze(1) - age_list.unsqueeze(0))
        dist_weight = dist_weight / dist_weight.max() # 归一化
        
        # 计算特征空间距离 (32, 5)
        it_dist = torch.cdist(img_feats, txt_feats, p=2)
        
        # 强制单调性：距离真实等级越远，特征距离必须越大
        L_cop = torch.mean(it_dist * dist_weight)
        return L_cop