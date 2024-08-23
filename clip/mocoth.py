import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import VisionTransformer

# 定义与预训练权重对应的模型结构
def create_model():
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    return model

# 只加载匹配的键
def load_weights(model, weights_path):
    state_dict = torch.load(weights_path, map_location='cpu')['state_dict']
    model_state_dict = model.state_dict()
    
    # 只加载匹配的权重
    matched_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    model_state_dict.update(matched_state_dict)
    
    model.load_state_dict(model_state_dict, strict=False)
    return model

# 检查是否完整加载
def check_load_status(model, state_dict):
    model_state_dict = model.state_dict()
    model_keys = set(model_state_dict.keys())
    weight_keys = set(state_dict.keys())
    
    missing_keys = model_keys - weight_keys
    unexpected_keys = weight_keys - model_keys
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    if not missing_keys and not unexpected_keys:
        print("All keys loaded successfully")

# 计算模型参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# 主程序
if __name__ == "__main__":
    # 模型和权重路径
    weights_path = '/root/lanyun-tmp/amu-jittor78/vit-b-300ep.pth.tar'
    
    # 创建模型
    model = create_model()
    
    # 加载权重
    model = load_weights(model, weights_path)
    
    # 检查是否完整加载
    state_dict = torch.load(weights_path, map_location='cpu')['state_dict']
    check_load_status(model, state_dict)
    
    # 计算模型参数量
    num_params = count_parameters(model)
    print(f"Total number of parameters: {num_params}")