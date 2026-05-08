import torch
import torch.nn as nn
from clip import clip 
# 【修复1】导入正确的类名 RankFormer
from models.rankformer import RankFormer 

class L2RPromptLearner(nn.Module):
    def __init__(self, clip_model, num_ranks, embeddings_dim):
        super().__init__()
        self.num_ranks = num_ranks
        
        # 【修改】使用共享前缀 + 等级特定后缀，保证tokenize后位置语义对齐
        context = "A medical ultrasound image of a liver"
        category_prompts = [
            f"{context} with normal tissue without fibrosis",
            f"{context} with mild fibrosis and portal expansion",
            f"{context} with moderate fibrosis extending beyond portal tracts",
            f"{context} with severe bridging fibrosis and distorted architecture",
            f"{context} with advanced cirrhosis and regenerative nodules"
        ]
        
        with torch.no_grad():
            tokens = clip.tokenize(category_prompts).to(next(clip_model.parameters()).device)
            initial_embeds = clip_model.token_embedding(tokens)

            # 动态计算实际最大 token 长度（避免硬编码 17 截断 EOT）
            seq_lens = []
            for i in range(tokens.shape[0]):
                nonzero = (tokens[i] != 0).nonzero(as_tuple=True)[0]
                if len(nonzero) > 0:
                    seq_lens.append(nonzero[-1].item() + 1)
                else:
                    seq_lens.append(1)
            seq_len = max(seq_lens)
            print(f"[L2RPromptLearner] Detected max token length: {seq_len}")

        # 保存 token IDs 供外部 TextEncoder 提取 EOT 特征
        self.psudo_sentence_tokens = tokens[:, :seq_len].clone()

        # 前 seq_len-1 个作为 context，最后 1 个作为 rank（避免截断）
        self.context_embeds = nn.Parameter(initial_embeds[:, :seq_len - 1, :].clone())
        self.rank_embeds = nn.Parameter(initial_embeds[:, seq_len - 1:seq_len, :].clone())

        real_embed_dim = initial_embeds.shape[-1]
        self.rankformer = RankFormer(embed_dim=real_embed_dim)

    def forward(self):
        R = torch.cat([self.context_embeds, self.rank_embeds], dim=1) 
        R_prime = self.rankformer(R)
        return R_prime