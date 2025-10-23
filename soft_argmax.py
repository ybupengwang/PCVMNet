import torch
import torch.nn as nn
import torch.nn.functional as F
# class SoftArgmax(nn.Module):
#     def __init__(self, window_size=50, temperature=0.1):
#         super().__init__()
#         self.window_size = window_size
#         self.temperature = temperature
#
#     def forward(self, heatmap):
#         B, C, H, W = heatmap.shape
#
#         # Step 1: 找 peak
#         heatmap_flatten = heatmap.view(B, C, -1)
#         max_indices = torch.argmax(heatmap_flatten, dim=-1)  # (B, C)
#         max_x = max_indices % W
#         max_y = max_indices // W
#
#         # Step 2: 计算局部 softmax 并填回整图
#         local_heatmap = torch.zeros_like(heatmap)
#         for b in range(B):
#             for c in range(C):
#                 # 窗口边界
#                 x0 = max(0, max_x[b, c] - self.window_size // 2)
#                 x1 = min(W, max_x[b, c] + self.window_size // 2 )
#                 y0 = max(0, max_y[b, c] - self.window_size // 2)
#                 y1 = min(H, max_y[b, c] + self.window_size // 2)
#
#                 patch = heatmap[b, c, y0:y1, x0:x1]
#                 patch = patch / self.temperature
#                 patch = F.softmax(patch.flatten(), dim=0).reshape_as(patch)
#
#                 local_heatmap[b, c, y0:y1, x0:x1] = patch
#
#         # Step 3: meshgrid for coordinates
#         xx, yy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
#         xx, yy = xx.to(heatmap.device), yy.to(heatmap.device)
#
#         x = (xx * local_heatmap).sum(dim=[2, 3]) / (W - 1)
#         y = (yy * local_heatmap).sum(dim=[2, 3]) / (H - 1)
#         coords = torch.stack([x, y], dim=-1)  # (B, C, 2)
#
#         # Step 4: Spread (方差)
#         x_diff = (xx[None, None, :, :] - x.unsqueeze(-1).unsqueeze(-1) * (W - 1)) ** 2
#         y_diff = (yy[None, None, :, :] - y.unsqueeze(-1).unsqueeze(-1) * (H - 1)) ** 2
#         spread = (x_diff + y_diff) * local_heatmap
#         spread = spread.sum(dim=[2, 3]).unsqueeze(-1)  # (B, C, 1)
#
#         # 可选：写入 spread 日志
#         with open("spread.txt", 'a') as f:
#             f.write('\t'.join(map(str, spread.view(-1).tolist())) + '\n')
#
#         return coords, spread

####这里是520备份的
class SoftArgmax(nn.Module):
    def __init__(self, window_size=16, temperature=0.01,sigma_factor=0.25):
        """
        Soft-Argmax 计算类，基于局部加权平均的方法从热图中提取关键点坐标。

        参数：
        - window_size: 选取的局部窗口大小
        - temperature: Softmax 温度参数，越小对峰值位置越敏感
        """
        super().__init__()
        self.window_size = window_size
        self.temperature = temperature
        self.sigma_factor = sigma_factor

    def forward(self, heatmap):
        """
        计算 Soft-Argmax，返回归一化的关键点坐标 (x, y)。

        参数：
        - heatmap: (B, C, H, W) 形状的热图

        返回：
        - coords: (B, C, 2) 形状的张量，表示关键点坐标
        - spread: (B, C, 1) 形状的张量，表示局部方差（不确定性）
        """
        B, C, H, W = heatmap.shape

        # Step 1: 找到每个关键点的最大值索引
        heatmap_flatten = heatmap.view(B, C, -1)  # (B, C, H*W)
        max_indices = torch.argmax(heatmap_flatten, dim=-1)  # (B, C)
        max_x = max_indices % W
        max_y = max_indices // W

        # Step 2: 局部加权平均（soft-argmax in local window）
        xx, yy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        xx = xx.to(heatmap.device).float()
        yy = yy.to(heatmap.device).float()

        coords = torch.zeros((B, C, 2), device=heatmap.device)
        spread = torch.zeros((B, C, 1), device=heatmap.device)
        cov = torch.zeros((B, C, 2, 2), device=heatmap.device)
        for b in range(B):
            for c in range(C):
                x0 = max_x[b, c].item()
                y0 = max_y[b, c].item()

                # 局部窗口范围
                xmin = max(x0 - self.window_size // 2, 0)
                xmax = min(x0 + self.window_size // 2, W)
                ymin = max(y0 - self.window_size // 2, 0)
                ymax = min(y0 + self.window_size // 2, H)

                # 提取局部窗口
                patch = heatmap[b, c, ymin:ymax, xmin:xmax]
                patch = patch / self.temperature
                patch = F.softmax(patch.view(-1), dim=0).view_as(patch)

                # 对应坐标窗口
                x_patch = xx[ymin:ymax, xmin:xmax]
                y_patch = yy[ymin:ymax, xmin:xmax]



                # 坐标加权平均
                x_mean = (x_patch * patch).sum()
                y_mean = (y_patch * patch).sum()
                coords[b, c, 0] = x_mean / (W - 1)
                coords[b, c, 1] = y_mean / (H - 1)

                # 计算方差（不确定性）
                x_diff = (x_patch - x_mean) ** 2
                y_diff = (y_patch - y_mean) ** 2
                var = ((x_diff + y_diff) * patch).sum()
                spread[b, c, 0] = var
                #计算协方差
                x_diff = x_patch - x_mean  # (window_h, window_w)
                y_diff = y_patch - y_mean

                var_xx = (patch * x_diff * x_diff).sum()
                var_yy = (patch * y_diff * y_diff).sum()
                cov_xy = (patch * x_diff * y_diff).sum()

                cov[b, c, 0, 0] = var_xx
                cov[b, c, 1, 1] = var_yy
                cov[b, c, 0, 1] = cov_xy
                cov[b, c, 1, 0] = cov_xy

        return coords, cov,spread
# class SoftArgmax(torch.nn.Module):
#     def __init__(self, window_size=80, temperature=0.01,sigma_factor=0.25):
#         super().__init__()
#         self.window_size = window_size
#         self.temperature = temperature
#
#     def forward(self, heatmap):
#         """
#         输入:
#             heatmap: (B, C, H, W) 关键点热图
#         输出:
#             coords: (B, C, 2) 关键点归一化坐标，范围[0,1]
#             covs: (B, C, 2, 2) 局部窗口加权协方差矩阵
#         """
#         B, C, H, W = heatmap.shape
#         pad = self.window_size // 2
#
#         # 找到最大值索引（关键点粗略位置）
#         heatmap_flat = heatmap.view(B, C, -1)
#         max_idx = heatmap_flat.argmax(dim=-1)  # (B,C)
#         max_x = max_idx % W
#         max_y = max_idx // W
#
#         # 对热图边缘进行padding方便提取局部窗口
#         heatmap_pad = F.pad(heatmap, (pad, pad, pad, pad), mode='constant', value=0)
#         heatmap_pad = heatmap_pad.view(B * C, 1, H + 2 * pad, W + 2 * pad)
#
#         # 使用unfold采集局部窗口patches
#         patches = heatmap_pad.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)
#         # patches形状 (B*C, 1, H, W, window_size, window_size)
#         patches = patches.squeeze(1).contiguous().view(B * C, H * W, self.window_size, self.window_size)
#
#         # 找到每个关键点对应的局部窗口patch
#         indices = (max_y * W + max_x).view(-1)  # (B*C)
#         local_patches = patches[torch.arange(B * C), indices, :, :]  # (B*C, window_size, window_size)
#
#         # 归一化局部窗口heatmap，温度缩放后softmax
#         local_patches = local_patches / self.temperature
#         weights = F.softmax(local_patches.view(B * C, -1), dim=-1).view(B * C, self.window_size, self.window_size)
#
#         # 构造局部窗口坐标网格，中心为0，范围[-pad, pad]
#         coord_range = torch.linspace(-pad, pad, self.window_size, device=heatmap.device)
#         xx, yy = torch.meshgrid(coord_range, coord_range, indexing='xy')  # (window_size, window_size)
#         xx = xx.unsqueeze(0)  # (1, window_size, window_size)
#         yy = yy.unsqueeze(0)
#
#         # 计算局部加权均值（偏移量）
#         x_mean = (weights * xx).sum(dim=(-2, -1))  # (B*C)
#         y_mean = (weights * yy).sum(dim=(-2, -1))  # (B*C)
#
#         # 关键点精细坐标 = 粗略坐标 + 局部偏移, 并归一化到0~1
#         x = (max_x.view(-1).float() + x_mean) / (W - 1)
#         y = (max_y.view(-1).float() + y_mean) / (H - 1)
#         coords = torch.stack([x, y], dim=-1).view(B, C, 2)
#
#         # 计算加权协方差矩阵
#         x_diff = xx - x_mean.unsqueeze(-1).unsqueeze(-1)  # (B*C, W, W)
#         y_diff = yy - y_mean.unsqueeze(-1).unsqueeze(-1)  # (B*C, W, W)
#
#         var_xx = (weights * x_diff * x_diff).sum(dim=(-2, -1))  # (B*C)
#         var_yy = (weights * y_diff * y_diff).sum(dim=(-2, -1))  # (B*C)
#         cov_xy = (weights * x_diff * y_diff).sum(dim=(-2, -1))  # (B*C)
#
#         covs = torch.zeros(B * C, 2, 2, device=heatmap.device)
#         covs[:, 0, 0] = var_xx
#         covs[:, 1, 1] = var_yy
#         covs[:, 0, 1] = cov_xy
#         covs[:, 1, 0] = cov_xy
#         covs = covs.view(B, C, 2, 2)
#
#         return coords, covs
