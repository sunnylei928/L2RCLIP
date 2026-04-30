import torch
import torch.nn as nn

class RankFormer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, hidden_dim=2048, alpha_init=0.01):
        super().__init__()
        # 物理锁：可学习的残差比例参数
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.ln = nn.LayerNorm(embed_dim)
        self.msa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, R):
        # 提取排序特征：f_FFN(f_MSA(f_LN(R)))
        x = self.ln(R)
        attn_out, _ = self.msa(x, x, x)
        extracted_features = self.ffn(attn_out)
        # 残差混合：R' = (1-alpha)*R + alpha*extracted
        return (1 - self.alpha) * R + self.alpha * extracted_features