import torch
import torch.nn as nn
from .rankformer import RankFormer
# 假设已经定义了 PlainPromptLearner 基类

class L2RPromptLearner(nn.Module):
    def __init__(self, clip_model, num_ranks, embeddings_dim):
        super().__init__()
        self.num_ranks = num_ranks
        self.embeddings_dim = embeddings_dim
        
        # 初始化可学习的 Context 和 Rank 嵌入 (参考 rank_prompt_learner.py)
        self.context_embeds = nn.Parameter(torch.randn(1, 16, embeddings_dim))
        self.rank_embeds = nn.Parameter(torch.randn(num_ranks, 1, embeddings_dim))
        
        # 植入 RankFormer
        self.rankformer = RankFormer(embed_dim=embeddings_dim)
        
        # 注册插值权重 (简化版)
        self.register_buffer("interpolation_weights", torch.eye(num_ranks))

    def forward(self):
        # 1. 生成基础文本特征 R
        # 简化逻辑：拼接 context 和 rank 嵌入
        ctx = self.context_embeds.expand(self.num_ranks, -1, -1)
        ranks = self.rank_embeds
        R = torch.cat([ctx, ranks], dim=1) # 假设 tail 位置
        
        # 2. RankFormer 强化排序关系 (R -> R')
        # 这里的 R 会经过自注意力机制，让 F0 到 F4 互相“对话”
        R_prime = self.rankformer(R)
        return R_prime