import torch
import torch.nn as nn

class RankFormer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, hidden_dim=2048, alpha_init=0.01):
        super().__init__()
        # 物理锁：控制自注意力特征注入的比例，初始值较小以保持预训练语义[cite: 9]
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.ln = nn.LayerNorm(embed_dim)
        self.msa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, R):
        # 1. 层归一化
        x = self.ln(R)
        # 2. 多头自注意力：让 F0-F4 在特征空间内互相“感知”排序[cite: 5, 9]
        attn_out, _ = self.msa(x, x, x)
        # 3. 前馈网络提取排序特征
        extracted_features = self.ffn(attn_out)
        # 4. 残差混合：保证模型不会完全跑偏[cite: 9]
        return (1 - self.alpha) * R + self.alpha * extracted_features