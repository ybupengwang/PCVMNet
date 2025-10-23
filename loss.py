import torch
import torch.nn as nn
from config import Config
# class JointsOHKMMSELoss(nn.Module):
#     def __init__(self, use_target_weight, topk=5):
#         super(JointsOHKMMSELoss, self).__init__()
#         self.criterion = nn.MSELoss(reduction='none')
#         self.use_target_weight = use_target_weight
#         self.topk = topk
#
#     def ohkm(self, loss):
#         ohkm_loss = 0.
#         for i in range(loss.size()[0]):
#             sub_loss = loss[i]
#             topk_val, topk_idx = torch.topk(
#                 sub_loss, k=self.topk, dim=0, sorted=False
#             )
#             tmp_loss = torch.gather(sub_loss, 0, topk_idx)
#             ohkm_loss += torch.sum(tmp_loss) / self.topk
#         ohkm_loss /= loss.size()[0]
#         return ohkm_loss
#
#     def forward(self, output, target, target_weight=1):
#         batch_size = output.size(0)
#         num_joints = output.size(1)
#         heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
#         heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
#
#         loss = []
#         for idx in range(num_joints):
#             heatmap_pred = heatmaps_pred[idx].squeeze()
#             heatmap_gt = heatmaps_gt[idx].squeeze()
#             if self.use_target_weight:
#                 loss.append(0.5 * self.criterion(
#                     heatmap_pred.mul(target_weight[:, idx]),
#                     heatmap_gt.mul(target_weight[:, idx])
#                 ))
#             else:
#                 loss.append(
#                     1 * self.criterion(heatmap_pred, heatmap_gt)
#                 )
#
#         loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
#         loss = torch.cat(loss, dim=1)
#
#         return self.ohkm(loss)
class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight=False, topk=13, num_joints=Config.point_num, ema_momentum=0.9):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk
        self.num_joints = num_joints
        self.ema_momentum = ema_momentum

        self.register_buffer("running_err", torch.zeros(num_joints))

    def forward(self, output, target, target_weight=1):
        batch_size = output.size(0)
        num_joints = output.size(1)

        # output, target: (B, C, H, W)
        # 拆 C 维，得到列表，每个元素是 [B, H, W]
        heatmaps_pred = torch.unbind(output, dim=1)  # len=19
        heatmaps_gt = torch.unbind(target, dim=1)    # len=19

        joint_losses = []

        for idx in range(num_joints):
            pred = heatmaps_pred[idx]  # [B, 800, 640]
            gt = heatmaps_gt[idx]      # [B, 800, 640]

            if self.use_target_weight:
                weight = target_weight[:, idx].view(batch_size, 1, 1)  # broadcast
                joint_loss = 0.5 * self.criterion(pred * weight, gt * weight)
            else:
                joint_loss = self.criterion(pred, gt)
                joint_loss = joint_loss

            joint_loss = joint_loss.mean(dim=(1, 2))  # → [B]
            joint_losses.append(joint_loss.unsqueeze(1))  # → [B, 1]

            # 更新历史误差
            with torch.no_grad():
                cur_mean = joint_loss.mean()
                self.running_err[idx] = (self.ema_momentum * self.running_err[idx] + (1 - self.ema_momentum) * cur_mean)

        # 拼接所有关节点误差 → [B, num_joints]
        loss = torch.cat(joint_losses, dim=1)

        # 选出历史上 topk 难学点
        _, hard_joint_idx = torch.topk(self.running_err, k=self.topk, largest=True)
        # if self.training :
        #     with open("hard_topk.txt", 'a') as f:
        #         # 写入表头
        #         f.write('\t'.join([str(i.item()) for i in hard_joint_idx]) + '\n')

        loss_hard = loss[:, hard_joint_idx]  # [B, topk]
        if self.topk ==19:
            return loss.mean()
        else:
            return loss_hard.mean()
class JointsOHKMCoorLoss(nn.Module):
    def __init__(self, use_target_weight=True, topk=5):
        super(JointsOHKMCoorLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.topk = topk
        self.criterion = nn.MSELoss(reduction='none')

    def ohkm(self, loss):
        # loss: [batch_size, num_joints]
        ohkm_loss = 0.
        for i in range(loss.size(0)):
            sample_loss = loss[i]  # [num_joints]
            topk_val, topk_idx = torch.topk(sample_loss, k=self.topk, dim=0, sorted=False)
            topk_loss = torch.gather(sample_loss, 0, topk_idx)
            ohkm_loss += torch.sum(topk_loss) / self.topk
        ohkm_loss /= loss.size(0)
        return ohkm_loss

    def forward(self, output, target, target_weight=None):
        # output & target: [batch_size, num_joints, 2]
        loss = self.criterion(output, target)  # shape: [B, J, 2]
        loss = torch.sum(loss, dim=2)  # 对x和y求和，shape: [B, J]

        if self.use_target_weight and target_weight is not None:
            # target_weight: [B, J, 1] 或 [B, J]
            if target_weight.dim() == 3:
                target_weight = target_weight.squeeze(-1)
            loss = loss * target_weight

        return self.ohkm(loss)



# class JointsOHKDiceLoss(nn.Module):
#     def __init__(self,  topk=5, num_joints=Config.point_num, ema_momentum=0.9, epsilon=1e-5):
#         super().__init__()
#
#         self.topk = topk
#         self.num_joints = num_joints
#         self.ema_momentum = ema_momentum
#         self.epsilon = epsilon
#
#         self.register_buffer("running_err1", torch.zeros(num_joints))
#
#     def forward(self, pred, target, target_weight=1):
#         """
#         pred, target: [B, C, H, W]
#         target_weight: [B, C]
#         """
#         B, C, H, W = pred.shape
#         pred = pred.view(B, C, -1)
#         target = target.view(B, C, -1)
#
#
#         intersection = 2 * (pred * target).sum(dim=2)
#         union = (pred ** 2).sum(dim=2) + (target ** 2).sum(dim=2)
#         dice = (intersection + self.epsilon) / (union + self.epsilon)
#         loss = 1 - dice  # [B, C]
#
#         # 平均到每个关键点上 → [C]
#         joint_mean_loss = loss.mean(dim=0)  # [C]
#
#         # 更新历史 EMA 误差
#         with torch.no_grad():
#             self.running_err1 = (
#                 self.ema_momentum * self.running_err1.to(joint_mean_loss.device) +
#                 (1 - self.ema_momentum) * joint_mean_loss.detach()
#             )
#
#         # 选择 topk 难学关键点
#         k = min(self.topk, self.num_joints)
#         _, hard_joint_idx = torch.topk(self.running_err1, k=k, largest=True)
#         _,bottom_idx = torch.topk(self.running_err1, k=k, largest=False)
#         selected_idx = torch.cat([hard_joint_idx, bottom_idx])
#         if self.training:
#             with open("hard_topkdice.txt", 'a') as f:
#                 f.write('\t'.join(map(str, selected_idx.tolist())) + '\n')
#
#         # 只优化 topk 个难点
#         loss_hard = loss[:, selected_idx]  # [B, k]
#         return 0.5*loss_hard.mean() + 0.5*joint_mean_loss.mean()



