import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal

# class StructureEncoder(nn.Module):
#     def __init__(self, in_dim=38, out_dim=256):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(in_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, out_dim)
#         )
#
#     def forward(self, struct_feat):  # (B, N, in_dim)
#         return self.encoder(struct_feat)  # (B, N, out_dim)

class StructureEncoder(nn.Module):
    def __init__(self, num_points=19):
        super().__init__()
        input_dim = num_points * num_points*2  # delta展开后维度

        hidden_dim = 256
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, delta_feat):
        # delta_feat: (B, N*N)

        correction = self.net(delta_feat)  # (B, N*N)
        return correction

#vae
# class StructureEncoder(nn.Module):
#     def __init__(self, input_dim=38, latent_dim=10, hidden_dim=256):
#         """
#         :param input_dim: 点集展平后的维度 (19个2D点 → 38)
#         :param latent_dim: 潜在空间维度
#         :param hidden_dim: 隐藏层维度
#         """
#         super().__init__()
#         # 编码器
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, latent_dim * 2)  # 输出μ和log_var
#         )
#         # 解码器
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim)
#         )
#
#     def reparameterize(self, mu, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def forward(self, x):
#         # 编码
#
#         h = self.encoder(x)  # (B, latent_dim*2)
#         mu, log_var = torch.chunk(h, 2, dim=-1)
#         z = self.reparameterize(mu, log_var)
#
#         # 解码
#         x_recon = self.decoder(z)
#         return x_recon, mu, log_var
#
#
# class VAEShapeConstraint(nn.Module):
#     def __init__(self, pretrained_path="best_vae.pth", latent_dim=10):
#         """
#         :param pretrained_vae_path: 预训练VAE的模型路径（若为None则新建VAE）
#         """
#         super().__init__()
#         self.vae = StructureEncoder(latent_dim=latent_dim)
#         self.vae.load_state_dict(torch.load(pretrained_path))
#         self.vae.requires_grad_(False)  # 冻结VAE
#
#         # 先验分布（标准正态）
#         self.prior = torch.distributions.Normal(
#             torch.zeros(latent_dim).to("cuda:0"),
#             torch.ones(latent_dim).to("cuda:0")
#         )
#
#     def forward(self, pred_points):
#         """
#         :param pred_points: 模型预测的点集 (B, 19, 2)
#         :return: 形状约束损失（KL散度）
#         """
#         # 展平点集并编码
#         x_pred = pred_points.view(pred_points.size(0), -1)  # (B, 38)
#         h = self.vae.encoder(x_pred)
#         mu, log_var = torch.chunk(h, 2, dim=-1)
#
#         # 计算后验分布与先验分布的KL散度
#         posterior = torch.distributions.Normal(mu, torch.exp(0.5 * log_var))
#         kl = torch.distributions.kl_divergence(posterior, self.prior).mean()
#         return kl

