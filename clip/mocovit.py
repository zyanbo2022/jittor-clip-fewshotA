import jittor as jt
import jittor.nn as nn
import numpy as np
import os
from mha import MultiheadAttention

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attn = MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def execute(self, x):
        attn_output, _ = self.attn(x, x, x)  # multihead attention
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=374):
        super(VisionTransformer, self).__init__()
        self.embed_dim = 768
        self.num_classes = num_classes
        self.patch_size = 16
        self.num_patches = (224 // self.patch_size) ** 2
        
        # Patch embedding
        self.projection = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Transformer layers
        self.transformer = nn.Sequential(*[
            TransformerEncoderLayer(self.embed_dim, num_heads=8, ff_hidden_dim=2048) for _ in range(12)
        ])
        
        # Classification head
        self.head = nn.Linear(self.embed_dim, num_classes)

    def execute(self, x):
        x = self.projection(x)  # Shape: (N, E, H, W)
        x = jt.flatten(x, 2).transpose(0, 2, 1)  # Flatten and transpose
        x = self.transformer(x)  # Apply transformer
        x = x.mean(dim=1)  # Pooling (mean)
        x = self.head(x)  # Classification head
        return x
        
def load_vit(pretrain_path):
    print("=> creating VisionTransformer model")
    model = VisionTransformer(num_classes=374)

    if os.path.isfile(pretrain_path):
        print("=> loading checkpoint '{}'".format(pretrain_path))
        state_dict = jt.load(pretrain_path)  # 直接加载state_dict

        # 手动加载匹配的参数
        for k, v in state_dict.items():
            if k in model.state_dict():
                model.state_dict()[k] = v

        # 手动加载参数
        model.load_parameters(model.state_dict())

        print("=> loaded pre-trained model '{}'".format(pretrain_path))
    else:
        print("=> no checkpoint found at '{}'".format(pretrain_path))
        raise FileNotFoundError
    
    # Replace the head with an identity layer
    model.head = nn.Identity()
    return model, 768  # Assuming ViT-Base, adjust output feature dimension accordingly

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
    
if __name__ == "__main__":
    jt.flags.use_cuda = 1
    model, dim = load_vit("/root/lanyun-tmp/amu-jittor78/vit-b-300ep.pkl")
    num_params = count_parameters(model)
    num_params_in_millions = num_params / 1_000_000
    print(f"Number of parameters in millions: {num_params_in_millions:.2f}M")
