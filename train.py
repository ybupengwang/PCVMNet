import torch
import time
from config import Config
Config = Config()
from test import get_errors
from utils import convert_to_true_coords
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
import torch
import torch.nn as nn
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: Tensor or list of shape [num_classes] 用于类别平衡，若为 None，则不使用
        gamma: focusing 参数，默认 2.0
        reduction: 'mean' | 'sum' | 'none'
        """
        super(FocalLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, list):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.alpha = alpha
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [B, C] 原始输出（未 softmax）
        targets: [B] 正确标签（整数索引）
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [B]
        pt = torch.exp(-ce_loss)  # [B] 计算 p_t
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # [B]

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            at = self.alpha[targets]  # [B]
            focal_loss *= at

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
def rle_loss_residual(pred, target, coarse_coord, scale_factor=0.001):
    """
    pred: (B, 19, 4) -> [:2] 是 Δmean， [2:] 是 σ
    target: (B, 19, 2) -> GT坐标
    coarse_coord: (B, 19, 2)
    """
    delta_mean = pred[:, :, :2]
    sigma = pred[:, :, 2:]  # 必须保证预测值通过 softplus/sqrt 等方式保证 > 0

    # 复原 refined 坐标
    mean = coarse_coord.detach() + delta_mean * scale_factor

    # log(sigma + 1) 以避免负值损失（比 log(sigma) 更稳定）
    log_term = torch.log(sigma + 1.0)
    squared_term = (target - mean) ** 2 / sigma

    loss = 0.5 * (log_term + squared_term)
    return loss.mean()

def heatmap_dice_loss_topk(pred, target, topk=13, epsilon=1e-5,ema_momentum=0.9,selected_idxs=[3, 11, 18],guding = False):
    """
    pred:  shape (N, C, H, W)
    target: shape (N, C, H, W)
    topk: 对每个样本取损失最大的 topk 个关键点计算平均损失
    """
    if guding:
        # 只选取指定关键点通道
        pred = pred[:, selected_idxs, :, :]
        target = target[:, selected_idxs, :, :]
    # flatten to (N, C, H*W)
    pred = pred.view(pred.shape[0], pred.shape[1], -1)
    target = target.view(target.shape[0], target.shape[1], -1)
    # calculate per-keypoint dice loss (N, C)
    intersection = 2 * (pred * target).sum(dim=2)
    union = (pred ** 2).sum(dim=2) + (target ** 2).sum(dim=2)

    dice = (intersection + epsilon) / (union + epsilon)
    loss = 1 - dice  # (N, C)

    # 对每个样本，选择 topk 个最大 loss（困难点）
    if topk >= loss.size(1):  # 如果 topk 超出点数，则退化为普通平均
        return loss.mean()
    topk_loss,hard_joint_idx = torch.topk(loss, topk, dim=1)  # shape (N, topk)

    # with open("hard_topkdice.txt", 'a') as f:
    #     f.write('\t'.join(map(str, hard_joint_idx.tolist())) + '\n')
    if guding:
        return 0.5 * topk_loss.mean() + 0.5 * loss.mean()
    if topk !=19:
        return topk_loss.mean()
    else:
        return  loss.mean()
def heatmap_dice_loss(pred, target, epsilon=1e-5):
    """
    pred:  shape (N, C, H, W)
    target: shape (N, C, H, W)
    """
    # flatten to (N, C, H*W)
    pred = pred.view(pred.shape[0], pred.shape[1], -1)
    target = target.view(target.shape[0], target.shape[1], -1)

    # calculate intersection and union
    intersection = 2 * (pred * target).sum(dim=2)
    union = (pred ** 2).sum(dim=2) + (target ** 2).sum(dim=2)

    dice = (intersection + epsilon) / (union + epsilon)
    loss = 1 - dice  # (N, C)

    return loss.mean()                    #如果单独训练的某个点的时候可以在这里进行修改


def generate_soft_label_tensor(labels, num_classes=6, sigma=1.0):
    """
    将整数等级标签（Tensor）转换为 soft label 分布。

    参数:
        labels: Tensor，形状 [B]，每个值是 [0, num_classes-1] 的整数标签
        num_classes: int，分类数
        sigma: float，高斯分布标准差

    返回:
        soft_labels: Tensor，形状 [B, num_classes]，每一行是 soft one-hot 标签分布
    """
    device = labels.device
    B = labels.size(0)

    # 创建 [num_classes] 的坐标轴 [0,1,2,...]
    x = torch.arange(num_classes, device=device).float()  # [C]
    x = x.unsqueeze(0).repeat(B, 1)  # [B, C]

    # 扩展 labels 的维度以便广播
    labels = labels.unsqueeze(1).float()  # [B, 1]

    # 高斯分布
    soft_labels = torch.exp(- (x - labels) ** 2 / (2 * sigma ** 2))  # [B, C]

    # 归一化成概率分布
    soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)

    return soft_labels

class KLDiscretLoss(nn.Module):
    def __init__(self):
        super(KLDiscretLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')  # 注意: 输入是 log_probs
        self.log_softmax = nn.LogSoftmax(dim=1)               # 对 model 输出做 log_softmax

    def forward(self, pred_logits, soft_labels):
        """
        pred_logits: [B, C] 模型输出的原始logits
        soft_labels: [B, C] soft one-hot标签 (每一行为概率分布, sum=1)
        """
        log_probs = self.log_softmax(pred_logits)  # [B, C]
        loss = self.criterion(log_probs, soft_labels)
        return loss

def train_model(model,soft_argmax, criterion, optimizer, scheduler, train_loader, test_loader, num_epochs,epoch, trans = Config.trans, struct_biaozhi = Config.struct_biaozhi, cvm_biaozhi = Config.cvm_biaozhi):
    print('Epoch{}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    focalloss= FocalLoss()
    model.train()
    loss_heat_total = 0
    loss_coo_total =0
    loss_coo_tran_total = 0
    loss_kl_total = 0
    dice_toal = 0
    loss_total = 0
    start_time = time.time()
    for i, (img, img_w,img_h,heatmaps, heatmaps_refine,  img_name, x_all, y_all, gt_x,gt_y,heatmaps_hrnet,cvm_target,img_yuanshi) in enumerate(train_loader):
        img = img.cuda(Config.GPU)
        img_h = img_h.cuda(Config.GPU)
        img_w = img_w.cuda(Config.GPU)
        heatmaps = heatmaps.cuda(Config.GPU)
        heatmaps_refine = heatmaps_refine.cuda(Config.GPU)
        cvm_target = cvm_target.cuda(Config.GPU)
        #软标签
        # cvm_target = generate_soft_label_tensor(cvm_target)
        #软标签
        img_yuanshi = img_yuanshi.cuda(Config.GPU)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs, outputs_refine,output_struct,refine_hp_2,refined_coords,log_sigma,feature_tran32,kl_loss,delta,coarse_coord,cvmclass= model(img,img_yuanshi)
            loss_heat = criterion(outputs, heatmaps)   #热图损失
            loss_heat_dice = heatmap_dice_loss_topk(outputs, heatmaps)  # 热图dice损失


            #第一个微调
            # loss_refine = criterion(outputs_refine, heatmaps_refine)
            # loss_refine_heat_dice = heatmap_dice_loss(outputs_refine, heatmaps_refine)#微调dice损失

            #hrnet 输出
            # loss_hrnet = criterion(feats,heatmaps_hrnet)
            # loss_hrnet = torch.mean(loss_hrnet)
            # loss_heatmaps_hrnet_dice = heatmap_dice_loss(feats, heatmaps_hrnet)

            #坐标回归torch.tensor([Config.scal_w,Config.scal_h])
            batch_size, num_landmarks, height, width = heatmaps.shape
            true_coords = convert_to_true_coords(gt_x,gt_y).cuda(Config.GPU)
            true_coords[:, :, 0] /= (img_w[:,None]-1)  #
            true_coords[:, :, 1] /= (img_h[:,None]-1)  #归一化坐标
            if cvm_biaozhi:
                # cvm_loss = focalloss(cvmclass, cvm_target)
                #软标签
                # loss_fn = KLDiscretLoss()
                # cvm_loss = loss_fn(cvmclass, cvm_target)
                #软标签
                cvm_loss = F.cross_entropy(cvmclass, cvm_target)

            #结构特征损失
            if struct_biaozhi:

                with torch.no_grad():
                    delta_gt = true_coords[:, :, None, :] - true_coords[:, None, :, :]  # (B, N, N, 2)
                    delta_gt_feature = delta_gt.view(delta_gt.size(0), -1)  # (B, N*N*2)
                struct_loss = F.mse_loss(output_struct,delta_gt_feature)
            else:
                struct_loss = torch.tensor(0.0)
            #专家开启 微调坐标
            if trans:
                #loss_coo_tran = F.mse_loss(refined_coords,true_coords)
                loss_coo_tran = rle_loss_residual(delta, true_coords,coarse_coord)
            else:
                loss_coo_tran =kl_loss= torch.tensor(0.0)
            # loss = loss_heat_dice
            loss =  loss_heat_dice +   cvm_loss  #注意test坐标
            #loss =  cvm_loss  # 注意test坐标
            #各类损失和
            loss_total += loss.item()
            loss_heat_total +=loss_heat.item() #第一热图mse损失
            loss_coo_total += loss_coo_tran.item() #专家微调损失
            loss_coo_tran_total += kl_loss.item() #专家平衡损失
            loss_kl_total +=cvm_loss.item()  #结构形状一致性损失
            dice_toal += loss_heat_dice.item()  #第一热图dice损失
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # 梯度裁剪（防止爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    loss_avg = loss_total / len(train_loader)
    loss_heat_avg = loss_heat_total/len(train_loader)
    loss_coo_avg = loss_coo_total/len(train_loader)
    loss_coo_tran_avg = loss_coo_tran_total/len(train_loader)
    loss_kl_avg = loss_kl_total/len(train_loader)
    dice_avg = dice_toal/len(train_loader)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'训练 Loss: {loss_avg:.6f} 热图loss:{loss_heat_avg:.6f} dice_avg{dice_avg:.4f} 专家微调坐标{loss_coo_avg:.6f} 结构形状{loss_kl_avg:.6f} 专家平衡{loss_coo_tran_avg:.6f}  用时：{elapsed_time:.4f}')
    get_errors(model, soft_argmax,test_loader, Config.test_gt_dir1, Config.save_results_path)
    scheduler.step()
