# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class MCP(nn.Module):
#     def __init__(self, in_channels=256, out_channels=128, spatial_size=32, lambda_weight=1.0):
#         """
#         Multi-View Complementary Prompter Module.
#
#         Args:
#             in_channels (int): 输入 token 的通道数（例如 Hl 和 Al）
#             out_channels (int): 输出 guiding prompt 的通道数
#             spatial_size (int): token 的空间大小 (e.g., 32 表示 32x32)
#             lambda_weight (float): 空间注意力加权参数 λ
#         """
#         super(MCP, self).__init__()
#         self.lambda_weight = nn.Parameter(torch.tensor(lambda_weight, dtype=torch.float32))
#
#         # 1×1 convs for g1 and g2
#         self.g1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.g2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#         # g3: 最后生成 guiding prompt
#         self.g3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
#
#         # softmax 用于计算空间注意力
#         self.softmax = nn.Softmax(dim=-1)
#
#         # 存储空间尺寸
#         self.spatial_size = spatial_size
#
#     def forward(self, H_l, A_l):
#         """
#         Args:
#             H_l (Tensor): 表情特征 [B, C, H, W]
#             A_l (Tensor): 关键点特征 [B, C, H, W]
#
#         Returns:
#             P_{l+1} (Tensor): guiding prompt [B, C_out, H, W]
#         """
#         # Step 1: 投影到低维空间
#         M_H = self.g1(H_l)  # [B, C_out, H, W]
#         M_A = self.g2(A_l)  # [B, C_out, H, W]
#
#         B, C, H, W = M_H.shape
#         assert H == self.spatial_size and W == self.spatial_size, "spatial size mismatch"
#
#         # Step 2: 空间注意力
#         MH_flat = M_H.view(B, C, -1)  # [B, C, H*W]
#         attn = self.softmax(MH_flat / self.lambda_weight)  # [B, C, H*W]
#         attn = attn.view(B, C, H, W)
#
#         M_H_weighted = M_H * attn  # [B, C, H, W]
#
#         # Step 3: 融合得到 guiding prompt
#         P_next = self.g3(M_H_weighted + M_A)  # [B, C_out, H, W]
#
#         return P_next
#
#
# if __name__ == "__main__":
#     # 模拟两个输入特征图（表情特征 Hl 和关键点特征 Al）
#     B, C_in, H, W = 4, 256, 32, 32
#     C_out = 128
#
#     Hl = torch.randn(B, C_in, H, W).cuda()
#     Al = torch.randn(B, C_in, H, W).cuda()
#
#     mcp = MCP(in_channels=256, out_channels=128, spatial_size=32).cuda()
#     guiding_prompt = mcp(Hl, Al)
#
#     print("Guiding Prompt shape:", guiding_prompt.shape)  # 应为 [4, 128, 32, 32]

import torch
import torch.nn as nn
import torch.nn.functional as F

class MCP_Token(nn.Module):
    def __init__(self, dim=768, proj_dim=256):
        """
        MCP module operating on token sequences.

        Args:
            dim: 输入 token 的通道维度
            proj_dim: 投影后的维度 D'
        """
        super(MCP_Token, self).__init__()
        self.lambda_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.g1 = nn.Linear(dim, proj_dim)  # 表情特征降维
        self.g2 = nn.Linear(dim, proj_dim)  # 关键点特征降维
        self.g3 = nn.Linear(proj_dim, dim)  # 输出 guiding prompt

    def forward(self, H_l, A_l):
        """
        Args:
            H_l: 表情 tokens, [B, N, D]
            A_l: 地标 tokens, [B, N, D]

        Returns:
            P_{l+1}: guiding prompt, [B, N, D]
        """
        # Step 1: 投影
        M_H = self.g1(H_l)  # [B, N, D']
        M_A = self.g2(A_l)  # [B, N, D']

        # Step 2: 计算注意力因子
        attn = F.softmax(M_H / self.lambda_weight, dim=1)  # [B, N, D']
        M_H_weighted = M_H * attn  # 空间加权 [B, N, D']

        # Step 3: 融合并回投影
        P_next = self.g3(M_H_weighted + M_A)  # [B, N, D]
        return P_next

if __name__ == "__main__":
    B, N, D = 2, 1024, 768
    H_l = torch.randn(B, N, D)
    A_l = torch.randn(B, N, D)

    mcp = MCP_Token(dim=768, proj_dim=256)
    P_next = mcp(H_l, A_l)

    print("Output guiding prompt shape:", P_next.shape)  # [2, 1024, 768]