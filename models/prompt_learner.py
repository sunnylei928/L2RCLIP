import torch
import torch.nn as nn
from clip import clip 
# 【修复1】导入正确的类名 RankFormer
from models.rankformer import RankFormer 

class L2RPromptLearner(nn.Module):
    def __init__(self, clip_model, num_ranks, embeddings_dim):
        super().__init__()
        self.num_ranks = num_ranks
        
        category_prompts = [
            "normal liver tissue without fibrosis",                  
            "mild fibrosis with portal expansion",                  
            "moderate fibrosis extending beyond portal tracts",      
            "severe bridging fibrosis with distorted architecture",  
            "advanced cirrhosis with regenerative nodules"           
        ]
        
        with torch.no_grad():
            tokens = clip.tokenize(category_prompts).to(next(clip_model.parameters()).device)
            initial_embeds = clip_model.token_embedding(tokens)
            
        # 【修复2】保存 token IDs 供外部 TextEncoder 提取 EOT 特征
        self.psudo_sentence_tokens = tokens[:, :17].clone() 
            
        self.context_embeds = nn.Parameter(initial_embeds[:, :16, :].clone())
        self.rank_embeds = nn.Parameter(initial_embeds[:, 16:17, :].clone())
        
        # 【修复1】实例化正确的类名
        real_embed_dim = initial_embeds.shape[-1] 
        self.rankformer = RankFormer(embed_dim=real_embed_dim)

    def forward(self):
        R = torch.cat([self.context_embeds, self.rank_embeds], dim=1) 
        R_prime = self.rankformer(R)
        return R_prime