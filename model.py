
from collections import OrderedDict
from soft_argmax import SoftArgmax
from structure import StructureEncoder
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet121_Weights
import torch.nn.functional as F
from config import Config
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GATConv, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv
from HR.HRnet import HRNet
#from structure import VAEShapeConstraint
from torchvision.transforms.functional import resize
from vit.vit import vit_base_patch16_224_in21k
from torchvision.utils import save_image
import os
from resnet.resnet import resnet50
from swin.swin import swin_tiny_patch4_window7_224
import time
def crop_patch_from_tensor(img_tensor, coords, margin=50, target_size=224):
    """
    img_tensor: (B, C, H, W)
    coords: (B, 13, 2) 归一化坐标
    return: patches (B, C, target_size, target_size)
    """
    B, C, H, W = img_tensor.shape
    patches = []

    for i in range(B):
        xs = (coords[i, :, 0] * W).long()
        ys = (coords[i, :, 1] * H).long()

        x_min = torch.clamp(xs.min() - margin, 0, W)
        x_max = torch.clamp(xs.max() + margin, 0, W)
        y_min = torch.clamp(ys.min() - margin, 0, H)
        y_max = torch.clamp(ys.max() + margin, 0, H)
        x_min = 1090
        x_max = 1659
        y_min = 1356
        y_max = 2132

        patch = img_tensor[i:i+1, :, y_min:y_max, x_min:x_max]
        patch = F.interpolate(patch, size=(224, 224), mode='bilinear', align_corners=False)  #默认尺寸为320*240
        #存一下图片看看
        # patch_img = patch.squeeze(0).cpu().detach()
        # timestamp = int(time.time() * 1000)
        # save_path = os.path.join(r"D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\patch", f"patch_{timestamp}_{i}.png")
        # save_image(patch_img, save_path)
        #存照片结束
        patches.append(patch)
        # 拼接成 (B, C, H, W)
    patches = torch.cat(patches, dim=0)
    # 一次性归一化
    # mean = torch.tensor([0.485, 0.456, 0.406], device=patches.device).view(1, 3, 1, 1)
    # std = torch.tensor([0.229, 0.224, 0.225], device=patches.device).view(1, 3, 1, 1)
    # patches = (patches - mean) / std
    return patches

def build_graph_from_landmarks():
    # coords: Tensor [13, 2]
    edge_list = [
        (0, 1), (1, 2),(2, 0),
        (3, 4), (4, 7),(7, 6), (6, 5),(5, 3),
        (8, 9),(9,12), (12,11),(11,10), (10,8)
    ]
    edge_list = edge_list + [(j, i) for (i, j) in edge_list]  # 添加反向边
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # [2, E]
    return edge_index

# class LandmarkGCN(nn.Module):
#     def __init__(self, in_dim=2, hidden_dim=64, num_classes=6):
#         super().__init__()
#         self.conv1 = GCNConv(in_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.classifier = nn.Linear(hidden_dim, num_classes)
#
#     def forward(self, x, edge_index, batch):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         x = global_mean_pool(x, batch)  # 每个图取平均
#         return self.classifier(x)
class LandmarkGCN(nn.Module):
    # def __init__(self, in_dim=2, hidden_dim=64, num_classes=6):
    #     super().__init__()
    #     nn1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
    #     nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
    #     self.conv1 = GINConv(nn1)
    #     self.conv2 = GINConv(nn2)
    #     self.classifier = nn.Linear(hidden_dim, num_classes)
    #
    # def forward(self, x, edge_index, batch):
    #     x = F.relu(self.conv1(x, edge_index))
    #     x = F.relu(self.conv2(x, edge_index))
    #     x = global_mean_pool(x, batch)
    #     return self.classifier(x)
    def __init__(self, num_classes=6):
        super().__init__()
        self.coord_encoder = nn.Linear(2, 64)

        self.conv1 = GATConv(64, 128, edge_dim=2)
        self.conv2 = GATConv(128, 64, edge_dim=2)

        self.curve_fc = nn.Linear(13 * 2, 64)

        self.classifier = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, data):
        # 编码坐标
        x = self.coord_encoder(data.x)  # (N, 64)

        # 图注意力卷积
        x = self.conv1(x, data.edge_index, edge_attr=data.edge_attr).relu()
        x = self.conv2(x, data.edge_index, edge_attr=data.edge_attr)

        # 图特征池化 (每个图单独聚合)
        x_graph = global_max_pool(x, data.batch)  # (B, 64)

        # 计算每个图的几何特征
        B = data.ptr.size(0) - 1
        x_curve = torch.stack([
            data.x[data.ptr[i]:data.ptr[i+1]].reshape(-1) for i in range(B)
        ], dim=0)  # (B, 13*2)
        x_curve = self.curve_fc(x_curve).relu()  # (B, 64)

        # 合并分类
        x_combined = torch.cat([x_graph, x_curve], dim=1)
        return self.classifier(x_combined)  # (B, num_classes)


# class FinetuneEncoder(nn.Module):
#     def __init__(self, in_channels=128, d_model=128, nhead=8, num_layers=4, num_experts=4,num_queries=19):
#         super().__init__()
#         self.d_model = d_model
#         self.num_queries = num_queries
#         self.num_experts = num_experts
#         # 可学习的 landmark content query 向量
#         self.landmark_query = nn.Parameter(torch.randn(1, num_queries, d_model))
#         # 特征对齐层（解决通道维度不匹配）
#         self.feat_proj = nn.Conv2d(in_channels, d_model, 1)
#         # Positional Encoding
#         self.pos_encoder = PositionalEncoding(d_model)
#         self.experts = nn.ModuleList([
#             TransformerExpert(d_model, nhead,num_layers=12) for _ in range(num_experts)
#         ])
#         # 门控网络输入为 [sampled_feat + spread(1)]
#         self.gate_net = nn.Sequential(
#             nn.LayerNorm(2*d_model),
#             nn.Linear(2*d_model, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_experts)
#         )
#
#         self.mu_head = nn.Sequential(
#             nn.Linear(d_model, 2),  # 输出x, y
#             nn.Tanh()  # 映射到 [-1, 1]
#         )
#         self.log_var = nn.Linear(d_model, 1)
#         self.spread_proj = nn.Linear(1, d_model)
#
#     def forward(self, fused_feature, coarse_coords,coarse_spread,feature_tran32):
#         """
#         Args:
#             fused_feature: 融合特征图 (B, C, H, W)
#             coarse_coords: 粗定位坐标 (B, K, 2)
#         Returns:
#             refined_coords: 精调坐标 (B, K, 2)
#             log_sigma: 分布参数 (B, K)
#         """
#         B, K = coarse_coords.shape[:2]
#         #coarse_coords = coarse_coords.detach()
#         device = fused_feature.device
#
#         # 1. 特征通道对齐（128->128）
#         projected_feat = self.feat_proj(fused_feature)  # (B,128,800,640)
#
#         # 2. 特征采样（论文3.3节公式6）
#         grid = coarse_coords.unsqueeze(2)  # (B,K,1,2)
#         sampled_feat = F.grid_sample(
#             projected_feat,
#             grid,
#             align_corners=False,
#             mode='bilinear'
#         ).squeeze(3).permute(0, 2, 1)  # (B,K,128)
#         # 5. 图像上下文查询生成（论文图3b）
#         flattened_feature = feature_tran32.flatten(2).permute(0, 2, 1)  # (B,128,H*W)  32*32 ==== B,32*32,128
#         landmark_query = self.landmark_query + sampled_feat   #将粗略定位特征与查询向量相加
#         x = torch.cat([landmark_query,flattened_feature], dim=1)  #(b,32*32+19,128)
#         # 加位置编码
#         x = self.pos_encoder(x)
#         # MoE门控路由计算
#         spread_feat = self.spread_proj(coarse_spread) #（B,K,d）
#         #gate_input = spread_feat+coarse_spread
#         gate_input = torch.cat([sampled_feat, spread_feat], dim=-1)  # (B, K, 2d)
#         gate_logits = self.gate_net(gate_input)  # (B, K, E)
#         topk = 1
#         topk_vals, topk_idx = torch.topk(gate_logits, topk, dim=-1)  # (B, K, topk)
#         #print(topk_vals, topk_idx)
#         topk_weights = F.softmax(topk_vals, dim=-1)  #  softmax over top-k logits
#         # 准备专家输入
#         landmark_input = x[:, :K, :]  # (B, K, d)
#         outputs = torch.zeros_like(landmark_input)
#         for i in range(topk):
#             expert_ids = topk_idx[:, :, i]  # (B, K)
#             weight = topk_weights[:, :, i]  # (B, K)
#
#             for e_idx, expert in enumerate(self.experts):
#                 # 找出哪些 (B, K) token 选择了当前专家
#                 mask = (expert_ids == e_idx).float().unsqueeze(-1)  # (B, K, 1)
#                 if mask.sum() == 0:
#                     continue  # 如果当前专家没有被选中，就跳过
#                 expert_out = expert(x)[:, :K, :]  # (B, K, d)
#                 outputs += expert_out * (weight.unsqueeze(-1) * mask)
#                 outputs = landmark_input+outputs
#
#
#         mu = self.mu_head(outputs)  # (B, K, 2)
#         log_var = self.log_var(outputs)
#
#         #print(mu)
#         #sigma = F.softplus(self.sigma_head(outputs)) + 1e-6  # [B, K, 1]
#         #delta_xy = (mu + 1) / 2
#         coords = coarse_coords + mu*0.02
#         coords = (coords+1)/2
#         coords = torch.clamp(coords, min=0.0, max=1.0)
#         return coords, log_var # 返回最终修正后的坐标 和 sigma

def extract_feat_at_coords(feat_map, coords):
    """
    feat_map: (B, C, H, W)
    coords: (B, N, 2), 归一化坐标 (0~1), 表示 coarse 粗略坐标
    return: (B, N, C) 每个点的特征向量
    """
    # B, C, H, W = feat_map.shape
    # N = coords.shape[1]

    # 将 coords 从 (0~1) 归一化到 (-1~1) 区间，适配 grid_sample
    coords_norm = coords.clone()
    coords_norm[..., 0] = coords_norm[..., 0] * 2 - 1  # x
    coords_norm[..., 1] = coords_norm[..., 1] * 2 - 1  # y

    # grid_sample 要求 grid shape 为 (B, H_out, W_out, 2)，我们设置为 (B, N, 1, 2)
    grid = coords_norm.unsqueeze(2)  # (B, N, 1, 2)

    # 提取对应点的特征 (B, C, N, 1)
    sampled_feat = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=True)  # (B, C, N, 1)

    # squeeze 掉多余维度 -> (B, N, C)
    sampled_feat = sampled_feat.squeeze(-1).permute(0, 2, 1)  # (B, N, C)

    return sampled_feat

#专家模型
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Sequential(

            nn.Linear(hidden_dim, 2), # 输出 refined 坐标
            nn.Tanh()
        )
        self.var_head = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Softplus()  # 保证方差为正
        )

    def forward(self, x):
        features = self.shared_net(x)
        mean = self.mean_head(features)
        var = self.var_head(features)
        return torch.cat([mean, var], dim=-1)


def stable_rle_loss(pred, target):
    mean = pred[:, :2]
    sigma = pred[..., 2:]
    var = sigma ** 2
    # 改进的核心部分
    log_term = torch.log(sigma + 1.0)  # 避免log(0)→-∞
    squared_term = (target - mean) ** 2 / var

    loss = 0.5 * (log_term + squared_term)
    return loss.mean()
#moe
class MoE(nn.Module):
    def __init__(self, input_dim, num_experts=4,top_k=1,temperature=1):
        super().__init__()
        self.experts = nn.ModuleList([Expert(input_dim) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.coord_norm = nn.LayerNorm(2)  # 对坐标单独归一化
        self.cov_norm = nn.LayerNorm(4)  # 对协方差单独归一化
        self.gate_norm = nn.LayerNorm(6)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 64),   #消融协方差
            nn.ReLU(),
            nn.Linear(64, num_experts)

        )

    def forward(self,x,coarse_coord):  # x: (B, 19, D) x的组成是μΣ和粗坐标特征组成
        coord_feat =x[:, :, :2]
        cov_feat = x[:, :, 2:6]
        gate_input = torch.cat([coord_feat, cov_feat], dim=-1)
        # gate_input = self.gate_norm(gate_input)
        gate_logits = self.gate(x)  # (B, 19, E)   #在这里可以改变门控输入特征长度
        # 温度调整 + softmax
        gate_weight = F.softmax(gate_logits / self.temperature, dim=-1)  # (B, 19, E)
        #负载均衡
        # load = gate_weight.sum(dim=(0, 1))
        # load_mean = load.mean()
        # loss = ((load - load_mean) ** 2).mean()
        #负载均衡结束
        #信息熵
        entropy = - (gate_weight * torch.log(gate_weight + 1e-8)).sum(dim=-1)  # per sample
        loss = entropy.mean()  # batch mean

        # --- Top-k gating ---
        topk_vals, topk_idx = torch.topk(gate_weight, self.top_k, dim=-1)  # (B, 19, top_k)
        # 打印专家选择
        # first_sample = topk_idx[0]
        # first_sample = first_sample.squeeze(-1)
        # print(first_sample.tolist())

        topk_weight = topk_vals / (topk_vals.sum(dim=-1, keepdim=True))  # (B, 19, top_k)
        # Compute expert outputs
        expert_outs = [expert(x) for expert in self.experts]  # list of (B, 19, 2)
        expert_outs = torch.stack(expert_outs, dim=2)  # (B, 19, E, 2)
        # 计算每个位置 (B, 19) 上专家输出的方差


        # Gather top-k expert outputs
        topk_expert_outs = torch.gather(expert_outs, 2,topk_idx.unsqueeze(-1).expand(-1, -1, -1, 4))  # (B, 19, top_k, 2)
        # 分离 mean 和 log_sigma

        # 加权求和
        delta = torch.sum(topk_weight.unsqueeze(-1) * topk_expert_outs, dim=2)  # (B, 19, 4)
        # mean 和 sigma 拆开使用
        delta_mean = delta[:, :, :2]


        refined_coord = coarse_coord.detach() + delta_mean*0.002
        refined_coord = torch.clamp(refined_coord, 0.0, 1.0)
        return refined_coord,loss,delta



class wblock(nn.Module):
    def __init__(self, conv3or1, in_channel, out_channel, y_channel=0):
        super(wblock, self).__init__()
        self.conv3or1 = conv3or1
        self.conv11 = nn.Sequential(OrderedDict([
            ('conv11_1', nn.Conv2d(in_channel + y_channel, out_channel, kernel_size=1)),
            ('norm11_1', nn.BatchNorm2d(out_channel)),
            ('relu11_1', nn.ReLU(inplace=True))
        ]))
        self.conv33 = nn.Sequential(OrderedDict([
            ('conv11_1', nn.Conv2d(in_channel + y_channel, out_channel, kernel_size=3, stride=1, padding=1)),
            ('norm11_1', nn.BatchNorm2d(out_channel)),
            ('relu11_1', nn.ReLU(inplace=True))
        ]))

    def forward(self, x, y):
        x = torch.cat((x, y), 1)
        if self.conv3or1 == 1:
            x = self.conv11(x)
        elif self.conv3or1 == 3:
            x = self.conv33(x)
        return x


class Farnet(nn.Module):
    def __init__(self,trans = Config.trans,struct_biaozhi=Config.struct_biaozhi,cvm_biaozhi = Config.cvm_biaozhi):
        super(Farnet, self).__init__()
        weights = DenseNet121_Weights.DEFAULT
        self.features = models.densenet121(weights=weights).features



        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear')

        self.wblock1 = wblock(1, 512, 512, 256)
        self.wblock2 = wblock(1, 1024, 1024, 256)
        self.wblock3 = wblock(3, 2048, 1024, 256)

        self.w1_conv11_0 = nn.Sequential(OrderedDict([
            ('conv11_0', nn.Conv2d(3, 32, kernel_size=1)),
            ('norm11_0', nn.BatchNorm2d(32)),
            ('relu11_0', nn.ReLU(inplace=True)),
        ]))
        self.w1_conv33_01 = nn.Sequential(OrderedDict([
            ('conv11_1', nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)),
            ('norm11_1', nn.BatchNorm2d(512)),
            ('relu11_1', nn.ReLU(inplace=True)),
        ]))
        self.w1_conv11_1 = nn.Sequential(OrderedDict([
            ('conv11_1', nn.Conv2d(512, 256, kernel_size=1)),
            ('norm11_1', nn.BatchNorm2d(256)),
            ('relu11_1', nn.ReLU(inplace=True)),
        ]))

        self.w1_conv11_2 = nn.Sequential(OrderedDict([
            ('conv11_2', nn.Conv2d(1280, 256, kernel_size=1)),
            ('norm11_2', nn.BatchNorm2d(256)),
            ('relu11_2', nn.ReLU(inplace=True)),
        ]))
        self.w1_conv11_3 = nn.Sequential(OrderedDict([
            ('conv11_3', nn.Conv2d(768, 256, kernel_size=1)),
            ('norm11_3', nn.BatchNorm2d(256)),
            ('relu11_3', nn.ReLU(inplace=True)),
        ]))

        self.mid_conv11_1 = nn.Sequential(OrderedDict([
            ('conv11_4', nn.Conv2d(512, 256, kernel_size=1)),
            ('norm11_4', nn.BatchNorm2d(256)),
            ('relu11_4', nn.ReLU(inplace=True)),
        ]))

        self.w2_conv11_1 = nn.Sequential(OrderedDict([
            ('conv11_1', nn.Conv2d(1024, 256, kernel_size=1)),
            ('norm11_1', nn.BatchNorm2d(256)),
            ('relu11_1', nn.ReLU(inplace=True)),
        ]))
        self.w2_conv11_2 = nn.Sequential(OrderedDict([
            ('conv11_2', nn.Conv2d(1280, 256, kernel_size=1)),
            ('norm11_2', nn.BatchNorm2d(256)),
            ('relu11_2', nn.ReLU(inplace=True)),
        ]))
        self.w2_conv11_3 = nn.Sequential(OrderedDict([
            ('conv11_3', nn.Conv2d(768, 256, kernel_size=1)),
            ('norm11_3', nn.BatchNorm2d(256)),
            ('relu11_3', nn.ReLU(inplace=True)),
        ]))
        self.w2_conv11_4 = nn.Sequential(OrderedDict([
            ('conv11_4', nn.Conv2d(512+Config.point_num, 128, kernel_size=1)),
            ('norm11_4', nn.BatchNorm2d(128)),
            ('relu11_4', nn.ReLU(inplace=True)),
        ]))
        self.w2_conv11_5 = nn.Sequential(OrderedDict([
            ('conv11_4', nn.Conv2d(192+Config.point_num, 64, kernel_size=1)),
            ('norm11_4', nn.BatchNorm2d(64)),
            ('relu11_4', nn.ReLU(inplace=True)),
        ]))
        self.conv33_stride1 = nn.Sequential(OrderedDict([
            ('conv33_0', nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)),
            ('norm33_0', nn.BatchNorm2d(512)),
            ('relu33_0', nn.ReLU(inplace=True)),
        ]))
        self.conv33_stride2 = nn.Sequential(OrderedDict([
            ('conv33_0', nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)),
            ('norm33_0', nn.BatchNorm2d(1024)),
            ('relu33_0', nn.ReLU(inplace=True)),
        ]))
        self.conv33_stride3 = nn.Sequential(OrderedDict([
            ('conv33_0', nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)),
            ('norm33_0', nn.BatchNorm2d(2048)),
            ('relu33_0', nn.ReLU(inplace=True)),
        ]))
        self.conv_33_refine1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_33_refine2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_11_refine = nn.Conv2d(64, Config.point_num, kernel_size=1)
        self.conv_33_last1 = nn.Conv2d(96+Config.point_num, 96+Config.point_num, kernel_size=3, stride=1, padding=1)
        self.conv_11_last = nn.Conv2d(96+Config.point_num, Config.point_num, kernel_size=1)
        self.conv_11_refine2 = nn.Conv2d(128, Config.point_num, kernel_size=1)
        self.conv_33_refine3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_11_refine3 = nn.Conv2d(256, Config.point_num, kernel_size=1)
        self.upsample2_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample4_1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)



        #HRnet
        # self.backbone = HRNet(48, 19, 0.1)

        self.trans = trans
        self.struct_biaozhi = struct_biaozhi
        self.cvm_biaozhi = cvm_biaozhi
        self.soft_argmax = SoftArgmax(window_size=16, temperature=0.01)  # 修改局部softmax尺寸
        if struct_biaozhi:
            self.structure_encoder = StructureEncoder(num_points=Config.point_num)

        #transformer部分
        if trans:
            self.moe = MoE(input_dim =96+6+Config.point_num)   # 原先121

        if cvm_biaozhi:
            #图神经网络
            # self.cvm =LandmarkGCN()
            # self.edge_index = build_graph_from_landmarks()
            #图神经网络结束

            #vit
            self.vit_branch = vit_base_patch16_224_in21k()
            #vit 结束
            #resnet
            # self.resnet_branch = resnet50(num_classes=6)
            #resnet结束
            #swin
            # self.swin_branch = swin_tiny_patch4_window7_224()
            #swin 结束
            #crossvit

            #crossvit结束

    def forward(self, x,img_yuanshi):

        # 计算 w1_f0，并应用 Attention Mask
        x_yuanshi = x
        w1_f0 = self.w1_conv11_0(x) #x:b,3,800,640    w1_f0:32,800,640
        x = self.features[0](x) #64,400,320
        w1_f1 = x
        for i in range(1, 5):
            x = self.features[i](x)  #i=1 batchnorm2d(64) i=2 RELU i=3 maxpool2d(kernel_size=3, stride=2) x:256,200,160
        w1_f2 = x
        for i in range(5, 7):
            x = self.features[i](x)  #512,100,80
        w1_f3 = x
        for i in range(7, 9):
            x = self.features[i](x) #先256,50,40 ===》1024,50,40
        w1_f4 = x
        for i in range(9, 12):
            x = self.features[i](x) #512,25,20  ===》1024,25,20  ===》 1024,25,20
        # first upsample and concat
        x = self.w1_conv33_01(x)   #512,25,20
        x = self.w1_conv11_1(x)    #256,25,20
        w2_f5 = x
        x = self.upsample2(x)    #256,50,40
        x = torch.cat((x, w1_f4), 1) #1280,50,40
        # second upsample and concat
        x = self.w1_conv11_2(x) #256,50,40
        w2_f4 = x
        x = self.upsample2(x) #256,100,80
        x = torch.cat((x, w1_f3), 1) #768,100,80
        # third upsample and concat
        x = self.w1_conv11_3(x) #256,100,80
        w2_f3 = x
        x = self.upsample2(x) #256,200,160
        x = torch.cat((x, w1_f2), 1) #512,200,160

        x = self.mid_conv11_1(x) #256,200，160
        w3_f2 = x

        x = self.conv33_stride1(x) #512,100,80
        x = self.wblock1(x, w2_f3) #512,100,80
        w3_f3 = x
        x = self.conv33_stride2(x)#1024,50,40
        x = self.wblock2(x, w2_f4) #1024,50,40
        w3_f4 = x
        x = self.conv33_stride3(x)#2048,25,20
        x = self.wblock3(x, w2_f5)#1024,25,20
        x = self.w2_conv11_1(x) #256,25,20
        x = self.upsample2(x)
        x = torch.cat((x, w3_f4), 1)#1280,50,40
        x = self.w2_conv11_2(x) #256,50,40
        x = self.upsample2(x) #256,100,80
        x = torch.cat((x, w3_f3), 1) #768,100,80
        x = self.w2_conv11_3(x) #256,100,80
        #添加 refine_hp3
        refine_hp3 = self.conv_33_refine3(x)  # 256,100,80
        refine_hp3 = self.conv_11_refine3(refine_hp3)  # 19,100,80
        refine_hp_3 = refine_hp3
        x = self.upsample2(x) #256,200,160
        refine3_up = self.upsample2(refine_hp3)  # 19,200,160
        x = torch.cat((x, w3_f2,refine3_up), 1)#512,200,160
        x = self.w2_conv11_4(x) #128,200,160
        #添加refine_hp2
        M2 = self.conv_33_refine2(x) #128,200,160
        refine_hp2 = self.conv_11_refine2(M2)  # 19,200,160
        refine_hp_2 = refine_hp2
        x = self.upsample2(x)#128,400,320
        refine2_up = self.upsample2(refine_hp2)  # 19,400,320
        x = torch.cat((x, w1_f1,refine2_up), 1) #192+19，400,320  L1层最后的合并   加
        x = self.w2_conv11_5(x) #64,400,320

        refine_hp = self.conv_33_refine1(x) #64,400,320
        refine_hp = self.conv_11_refine(refine_hp) #19,400,320

        x = self.upsample2(x)   #64,800,640
        refine1_up = self.upsample2(refine_hp)# 19,800,640
        x = torch.cat((x, w1_f0, refine1_up), 1) #x：64，w1_f0：32  refine1_up：19  最后变成115,800,640
        # output
        hp = self.conv_33_last1(x) #115,800,640
        hp = self.conv_11_last(hp)  # 19,800,640   #预测的热图
        hp = torch.sigmoid(hp)

        #HRnet
        # hp,feat_hp = self.backbone(x)
        # refine_hp =refine_hp_2 = None


        coords, cov,_ = self.soft_argmax(hp)  # coords是0到1 spread不确定性
        if self.cvm_biaozhi:
            #图神经网络
            # batch_graphs = []
            # for i in range(coords.shape[0]):
            #     x = coords[i]
            #     data = Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr if hasattr(self, 'edge_attr') else None)
            #     batch_graphs.append(data)
            # batch = Batch.from_data_list(batch_graphs).to(coords.device)
            # cvmclass = self.cvm(batch)
            #图神经网络部分结束
            #vit部分
            # patches = crop_patch_from_tensor(img_yuanshi, coords)
            # cvmclass = self.vit_branch(patches)
            #vlp
            cvmclass = self.vit_branch(M2,hp)

            #vit部分结束

            #resnet部分
            # patches = crop_patch_from_tensor(img_yuanshi, coords)
            # cvmclass = self.resnet_branch(patches)
            #resnet结束

            #swin部分
            # patches = crop_patch_from_tensor(img_yuanshi, coords)
            # cvmclass = self.swin_branch(patches)
            #swin结束
        else:
            cvmclass = None


        #计算结构特征
        if self.struct_biaozhi:
            delta = coords[:, :, None, :] - coords[:, None, :, :]  # (B, N, N)
            delta_feat = delta.view(delta.size(0), -1)  # (B, N*N*2)
            recon = self.structure_encoder(delta_feat)  # 输入为 (B, N*N*2) # 预测结构向量
            kl_loss = None
        else:
            recon =kl_loss =None
        #transformer部分 forward部分
        if self.trans:
            sampled_feat = extract_feat_at_coords(x,coords)    #这里把x改成了feat_hp  切记坐标尺寸
            moe_input = torch.cat([coords,cov.reshape(cov.shape[0],cov.shape[1],-1),sampled_feat],dim=-1)
            refined_cooder,moe_loss,delta = self.moe(moe_input,coords)
            log_sigma = feature_tran32 = None
        else:
            refined_cooder=log_sigma=feature_tran32=moe_loss  = delta= None


        return hp, refine_hp, recon, refine_hp_2,refined_cooder,log_sigma,feature_tran32,moe_loss,delta,coords,cvmclass



# if __name__ == '__main__':
#     batch_size = 3
#     num_points = 19
#     input_dim = 32
#     x_sim = torch.randn(batch_size, num_points, input_dim)
#     moe = MoE(input_dim=input_dim, num_experts=3, top_k=2)
#     out = moe(x_sim)
#     print("Output shape:", out.shape)