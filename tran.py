import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FinetuneEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, num_queries=64):
        super().__init__()
        self.num_queries = num_queries

        # 初始化内容查询 (K x d_model)
        self.content_queries = nn.Embedding(num_queries, d_model)

        # Transformer编码层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 坐标偏移预测网络
        self.coord_refiner = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 3)  # 输出(Δx, Δy, logσ)
        )
        self.pos_proj = nn.Sequential(
           nn.Linear(128, d_model),
           nn.LayerNorm(d_model),
           nn.GELU()
   )

    def forward(self, fused_feature, coarse_coords):
        """
        Args:
            fused_feature: 融合特征图 (B, C, H, W)
            coarse_coords: 粗定位坐标 (B, K, 2)
        Returns:
            refined_coords: 精调坐标 (B, K, 2)
            sigma: 分布参数 (B, K)
        """
        batch_size = fused_feature.size(0)

        # 生成位置编码
        pos_embed = self.generate_positional_embedding(coarse_coords)  # (B, K, d_model)

        # 初始化内容查询
        content_queries = self.content_queries.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, K, d_model)

        # 特征图扁平化处理
        flattened_feature = fused_feature.flatten(2).permute(2, 0, 1)  # (H*W, B, C)

        # Transformer处理
        for layer in self.transformer.layers:
            # 自注意力阶段
            content_queries = layer(content_queries, pos_embed)

            # 交叉注意力阶段
            content_queries = layer(content_queries, memory=flattened_feature)

        # 坐标修正预测
        residuals = self.coord_refiner(content_queries)  # (B, K, 3)
        delta_xy = residuals[..., :2]
        log_sigma = residuals[..., 2]

        # 更新坐标
        refined_coords = coarse_coords + delta_xy

        return refined_coords, log_sigma

    def generate_positional_embedding(self, coords):
        """基于粗坐标生成位置编码"""
        # 将坐标归一化到[-1,1]
        normalized_coords = 2 * coords - 1

        # 使用傅里叶特征编码
        freq_bands = torch.linspace(1.0, 10.0, 64, device=coords.device)
        sin_features = torch.sin(freq_bands * math.pi * normalized_coords.unsqueeze(-1))
        cos_features = torch.cos(freq_bands * math.pi * normalized_coords.unsqueeze(-1))
        pos_embed = torch.cat([sin_features, cos_features], dim=-1)

        # 投影到d_model维度
        return self.pos_proj(pos_embed)  # (B, K, d_model)
