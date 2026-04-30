import os.path as osp  # 导入路径处理模块，简化路径操作

import torch  # 导入 PyTorch 库，支持张量操作和神经网络功能
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torchvision.models as models  # 导入 torchvision 中的预定义模型
from clip import clip  # 导入 CLIP 模型和相关工具

import sys
sys.path.insert(0, '/home/ubuntu/lq/OrdinalCLIP')

from ordinalclip.models import MODELS
from ordinalclip.models import image_encoders  # 导入自定义的图像编码器
from ordinalclip.models.builder import MODELS  # 导入用于注册和管理模型的工具
from ordinalclip.models.prompt_leaners import PROMPT_LEARNERS  # 导入提示学习器注册工具
from ordinalclip.models.prompt_leaners.plain_prompt_learner import PlainPromptLearner  # 导入简单的提示学习器

logger = get_logger(__name__)  # 初始化日志记录器



@MODELS.register_module()  # 将当前类注册为模型模块
class OrdinalCLIP(nn.Module):  # 继承 nn.Module，定义 OrdinalCLIP 模型
    def __init__(
        self,
        text_encoder_name,  # 文本编码器的名称
        image_encoder_name,  # 图像编码器的名称
        prompt_learner_cfg,  # 提示学习器的配置
        dropout_rate=0.5,
        **kwargs,  # 其他可选参数
    ) -> None:
        super().__init__()  # 调用父类的初始化方法

        if kwargs:  # 如果有额外的参数，输出日志
            logger.info(f"irrelevant kwargs: {kwargs}")

        # 加载 CLIP 模型到 CPU
        clip_model = load_clip_to_cpu(
            text_encoder_name,
            image_encoder_name,
            root=osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", ".cache", "clip"),
        )

        # 将模型转换为 float32 类型
        clip_model.float()
        logger.info("convert `clip_model` to float32. if need fp16 model, call `clip.model.convert_weights`")

        # 获取图像编码器和文本编码器
        self.image_encoder = clip_model.visual  # 图像编码器
        self.text_encoder = TextEncoder(clip_model)  # 文本编码器
        self.dropout = nn.Dropout(p=dropout_rate)
        # 设置提示学习器并初始化
        prompt_learner_cfg.update(dict(clip_model=clip_model))
        self.prompt_learner: PlainPromptLearner = PROMPT_LEARNERS.build(prompt_learner_cfg)
        self.psudo_sentence_tokens = self.prompt_learner.psudo_sentence_tokens  # 提示学习器生成的伪句子
        self.logit_scale = clip_model.logit_scale  # 用于计算最终相似度的缩放因子

        # 获取文本编码器的嵌入维度
        self.embed_dims = clip_model.text_projection.shape[1]
        # 获取 rank 数量
        # 【修复4】直接导入并使用你写的 L2RPromptLearner
        from models.prompt_learner import L2RPromptLearner
        
        # 直接实例化你的医学先验学习器
        self.num_ranks = prompt_learner_cfg.get("num_ranks", 5) # 默认 5 个等级 (F0-F4)
        self.prompt_learner = L2RPromptLearner(
            clip_model=clip_model,
            num_ranks=self.num_ranks,
            embeddings_dim=self.embed_dims
        )
        
        # 同步伪标签
        self.psudo_sentence_tokens = self.prompt_learner.psudo_sentence_tokens


    def forward(self, images):
        sentence_embeds = self.prompt_learner()  # 通过提示学习器获取句子嵌入
        psudo_sentence_tokens = self.psudo_sentence_tokens  # 获取伪句子标记
        text_features = self.text_encoder(sentence_embeds, psudo_sentence_tokens)  # 获取文本特征
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化文本特征

        # 获取图像特征并归一化
        image_features = self.image_encoder(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        image_features = self.dropout(image_features)
        
        logit_scale = self.logit_scale.exp()  # 计算缩放因子
        logits = logit_scale * image_features @ text_features.t()  # 计算图像和文本的相似度

        return logits, image_features, text_features  # 返回相似度、图像特征和文本特征

    # 此方法仅计算文本特征，适用于仅关心文本编码的场景。
    def forward_text_only(self):
        sentence_embeds = self.prompt_learner()  # 通过提示学习器获取句子嵌入
        psudo_sentence_tokens = self.psudo_sentence_tokens  # 获取伪句子标记
        text_features = self.text_encoder(sentence_embeds, psudo_sentence_tokens)  # 获取文本特征

        return text_features  # 返回文本特征


    def encode_image(self, x):
        return self.image_encoder(x) # 使用图像编码器获取图像特征

# 负责从 CLIP 模型中提取和处理文本特征。包括位置编码、通过 Transformer 生成文本特征，
# 以及使用投影层将其映射到嵌入空间。
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer  # 获取 CLIP 模型中的 transformer 部分
        self.positional_embedding = clip_model.positional_embedding  # 获取位置编码
        self.ln_final = clip_model.ln_final  # 获取最后的 LayerNorm 层
        self.text_projection = clip_model.text_projection  # 获取文本投影层

    def forward(self, prompts, tokenized_prompts):
        # 【修复3】动态获取输入 prompt 的长度 (这里是 17)
        seq_len = prompts.shape[1]
        
        # 截断位置编码以匹配长度
        pos_emb = self.positional_embedding[:seq_len, :].type(self.dtype)
        
        # 现在维度对齐了，可以安全相加
        x = prompts.type(self.dtype) + pos_emb  
        
        x = x.permute(1, 0, 2)  
        x = self.transformer(x)  
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype)  

        # 提取 eot 特征并通过文本投影层
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype  # 返回 transformer 所用的数据类型


# 加载 CLIP 模型的主函数，支持加载文本和图像编码器，处理模型路径，
# 支持自定义图像编码器或 torchvision 的标准模型。
def load_clip_to_cpu(
    text_encoder_name,
    image_encoder_name,
    root=osp.join(osp.expanduser("~/.cache/clip")),
):
    # 加载 CLIP 模型并返回
    print_func = logger.info if logger else print
    print_func("Building CLIP model...")
    text_backbone_name = text_encoder_name
    print_func(f"Text backbone : {text_backbone_name}'s counterpart.")
    url = clip._MODELS[text_backbone_name]
    model_path = clip._download(url, root=root)

    try:
        # 尝试加载 JIT 模型
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())  # 构建 CLIP 模型

    # 加载图像编码器
    embed_dim = model.text_projection.shape[1]
    input_resolution = model.visual.input_resolution
    image_backbone_name = image_encoder_name
    print_func(f"Image backbone: {image_backbone_name}")

    if image_backbone_name != text_backbone_name:
        # 加载自定义图像编码器或 torchvision 模型
        MODEL = getattr(image_encoders, image_backbone_name, None) or getattr(models, image_backbone_name, None)
        if MODEL is None:
            raise ValueError(f"Invalid torchvision model name: {image_backbone_name}")
        model.visual = MODEL(num_classes=embed_dim)  # 使用指定的图像模型
        model.visual.input_resolution = input_resolution
    else:
        print_func(f"CLIP Image encoder: {image_backbone_name}!")

    return model  # 返回加载的模型

