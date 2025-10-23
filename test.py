#import cv2
import numpy as np
import torch
#import xlwt
from config import Config
from utils import get_markposion_fromtxt, get_prepoint_from_htmp, get_prepoint_from_htmp_topk
from torchvision import transforms
from PIL import Image
Config = Config()
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import torch.nn.functional as F
import time
from data import get_final_preds
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report,confusion_matrix
import torchvision.transforms.functional as TF
def gaussian_blur1(hm, kernel):
    sigma = kernel / 6  # 经验公式：kernel=6*sigma 可近似匹配 OpenCV 效果
    border = (kernel - 1) // 2

    num_joints, height, width = hm.shape

    for j in range(num_joints):
        origin_max = np.max(hm[j])
        # 创建带边界的扩展热图
        dr = np.zeros((height + 2 * border, width + 2 * border), dtype=np.float32)
        dr[border: -border, border: -border] = hm[j].copy()
        # 使用 scipy 进行高斯模糊
        dr = gaussian_filter(dr, sigma=sigma)
        # 裁剪回原始大小
        cropped = dr[border: -border, border: -border]
        if np.max(cropped) > 0:
            hm[j] = cropped * (origin_max / np.max(cropped))
        else:
            hm[j] = cropped
    return hm
def refine_keypoints_dark_udp(keypoints: np.ndarray, heatmaps: np.ndarray,
                              blur_kernel_size: int) -> np.ndarray:
    """Refine keypoint predictions using distribution aware coordinate decoding
    for UDP. See `UDP`_ for details. The operation is in-place.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - heatmap size: [W, H]

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        heatmaps (np.ndarray): The heatmaps in shape (K, H, W)
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)

    .. _`UDP`: https://arxiv.org/abs/1911.07524
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    # modulate heatmaps
    heatmaps = gaussian_blur1(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 1e-3, 50., heatmaps)
    np.log(heatmaps, heatmaps)

    heatmaps_pad = np.pad(
        heatmaps, ((0, 0), (1, 1), (1, 1)), mode='edge').flatten()

    for n in range(N):
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(K, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints[n] -= np.einsum('imn,ink->imk', hessian,
                                  derivative).squeeze()

    return keypoints
def get_heatmap_maximum(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, (
        f'Invalid shape {heatmaps.shape}')

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)

    y_locs, x_locs = np.unravel_index(
        np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.] = -1

    if B:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)

    return locs, vals

def geometric_consistency_loss(coords, recon_delta):
    """
    coords: 当前预测的坐标 (B, N, 2)
    recon_delta: structure_encoder重建的结构向量 (B, N*N*2)
    """
    # 计算当前坐标的结构向量
    current_delta = coords[:, :, None, :] - coords[:, None, :, :]  # (B, N, N, 2)
    current_delta = current_delta.view(coords.size(0), -1)  # (B, N*N*2)

    # 约束重建的结构与当前结构一致
    loss = F.mse_loss(recon_delta, current_delta.detach())  # 只优化encoder

    # 可选：加入距离/角度等解剖约束
    # loss += distance_constraint(coords)
    return loss






# def visualize_landmark_heatmaps(outputs,num_landmarks=19,batch_idx=0,cols=5,save_dir='image',save_format='png',dpi=300):
#     # 创建存储目录
#     os.makedirs(save_dir, exist_ok=True)
#     # 时间戳用于区分不同运行批次
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     rows = (num_landmarks + cols - 1) // cols
#     fig = plt.figure(figsize=(15, rows * 3))
#     # 完整热图保存
#     full_heatmap_path = os.path.join(save_dir, f"full_{timestamp}_batch{batch_idx}.{save_format}")
#     for i in range(num_landmarks):
#         heatmap = outputs[batch_idx, i, :, :].cpu().detach().numpy()
#         # 创建子图
#         plt.subplot(rows, cols, i + 1)
#         img = plt.imshow(heatmap, cmap='jet')
#         plt.colorbar(img, fraction=0.046, pad=0.04)  # 优化色标显示比例
#         plt.title(f"Landmark {i + 1}", fontsize=9)  # 缩小标题字号
#         plt.axis('off')
#         # # 单独保存每个地标的热图
#         # landmark_path = os.path.join(save_dir,
#         #                              f"batch{batch_idx}_landmark{i + 1}_{timestamp}.{save_format}")
#         # plt.savefig(landmark_path, bbox_inches='tight', dpi=dpi)
#     # 保存完整组合图
#     plt.tight_layout()
#     plt.savefig(full_heatmap_path,bbox_inches='tight',dpi=dpi,facecolor='white')  # 设置背景为白色
#     plt.close(fig)

best_xiaoyu20 =best_xiaoyu201= 0.0  # 初始设为 0，或你可以设为一个更合理的起始值
best_RME = best_RME1 = 9999
best_acc = 0
results_dict = {}
def get_errors(model, soft_argmax,test_loader, note_gt_dir, save_path,trans=Config.trans):
    start_time = time.time()
    model.eval()
    # if Config.trans:
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     for param in model.structure_encoder.parameters():
    #         param.requires_grad = True
    model.cuda(Config.GPU)
    loss = np.zeros(Config.point_num)
    num_err_below_20 = np.zeros(Config.point_num)
    num_err_below_25 = np.zeros(Config.point_num)
    num_err_below_30 = np.zeros(Config.point_num)
    num_err_below_40 = np.zeros(Config.point_num)
    img_num = 0
    loss1 = np.zeros(Config.point_num)
    num1_err_below_20 = np.zeros(Config.point_num)
    num1_err_below_25 = np.zeros(Config.point_num)
    num1_err_below_30 = np.zeros(Config.point_num)
    num1_err_below_40 = np.zeros(Config.point_num)
    residuals= []
    spread = torch.zeros(Config.point_num, device=Config.GPU)
    global best_xiaoyu20,best_xiaoyu201,best_RME,best_RME1,best_acc
    save_model_path = "best_model_cvm2.pth"  # 设定保存权重的文件路径
    save_model_path1 = "best_moe_cvm2.pth"  # 设定保存权重的文件路径
    with open('zuobiao.txt', 'w') as f:
        f.truncate(0)

    cvm_preds = []
    cvm_targets = []
    for img_num, (img, img_w,img_h,heatmaps, heatmaps_refine, img_name, _, _,_,_,heatmaps_hrnet,cvm_target,img_yuanshi) in enumerate(test_loader):
        #print('图片', img_name[0])
        img = img.cuda(Config.GPU)
        img_w = img_w.cpu().numpy()
        img_h = img_h.cpu().numpy()
        img_yuanshi = img_yuanshi.cuda(Config.GPU)
        with torch.no_grad():
            outputs,_,_,_,refined_coords,log_sigma,_,_,_,_,cvmclass= model(img,img_yuanshi)
            # 热图输出
            #darkudp
            # heatmaps = outputs[0]
            # heatmaps = heatmaps.cpu().detach().numpy()
            # keypoints, scores = get_heatmap_maximum(heatmaps)
            # # unsqueeze the instance dimension for single-instance results
            # keypoints = keypoints[None]
            # scores = scores[None]
            # keypoints = refine_keypoints_dark_udp(
            #     keypoints, heatmaps, blur_kernel_size=11)
            #
            # #W, H = (Config.resize_w,Config.resize_h)
            # H = heatmaps.shape[1]
            # W = heatmaps.shape[2]
            # keypoints = keypoints / [W - 1, H - 1] * (1934,2399)
            # keypoints = np.round(keypoints).astype(int)
            # pred = keypoints[0]
            #darkudp结束
            #直接argmax
            #pred = get_prepoint_from_htmp(outputs[0].cpu().detach().numpy(), Config.scal_w, Config.scal_h,Config.resize_w,Config.resize_h)  # 热图转坐标
            #直接argmax结束
            #softmax开始
            zuobiao,cov,spread = soft_argmax(outputs)
            #softmax结束  选择这个

        #spread = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1)
        #利用泰勒展开求
        preds,maxvals = get_final_preds(outputs.cpu().detach().numpy())
        zuobiao = zuobiao.cpu().numpy()
        #refined_coords = refined_coords.cpu().numpy()
        # print(maxvals)

        preds[:, :, 0] = preds[:, :, 0] * (img_w[:,None]-1) / (Config.resize_w - 1)
        preds[:, :, 1] = preds[:, :, 1] * (img_h[:,None]-1) / (Config.resize_h - 1)
        pred = preds[0]   ##为了测试后处理方法效果
        #结束利用泰勒展开求

        # 翻转做一次求平均：
        # flipped_img = TF.hflip(img)
        # f_outputs, _, _, _, f_refined_coords, _, _, _ = model(flipped_img)
        # f_refined_coords[..., 0] = 1.0 - f_refined_coords[..., 0]
        # print(refined_coords[0,:,:])
        # print(f_refined_coords[0,:,:])
        # refined_coords = (refined_coords+f_refined_coords)/2

        if trans:
            # refined_coords = tta_optimize(model, refined_coords,fixed_indices=[6,7,8,10,13,14,16],n_iters=5, lr=0.0001)
            zuobiao = refined_coords

        zuobiao[:,:,0] = zuobiao[:,:,0]*(img_w[:,None]-1)
        zuobiao[:, :, 1] = zuobiao[:, :, 1] * (img_h[:,None]-1)
        zuobiao = zuobiao[0]

        #提取真实值
        note_gt_road = note_gt_dir + '/' + img_name[0].split('.')[0] + '.txt'
        # note_gt_road = 'D:/deeplearning/Anatomic-Landmark-Detection/process_data/cepha/antrainlabel' + '/' + img_name[0].split('.')[0] + '.txt'
        gt_x, gt_y,label= get_markposion_fromtxt(Config.point_num, note_gt_road)
        gt_x = np.trunc(np.reshape(gt_x, (Config.point_num, 1)))
        gt_y = np.trunc(np.reshape(gt_y, (Config.point_num, 1)))
        gt = np.concatenate((gt_x, gt_y), 1)

        #计算一下准确性
        pred_class = torch.argmax(cvmclass, dim=1).cpu()
        true_class = cvm_target.cpu()
        for a, b, c in zip(img_name, true_class, pred_class):
            if a not in results_dict:
                results_dict[a] = [str(b)]  # 第一个是 true label
            results_dict[a].append(str(c))  # 每轮追加预测值
        cvm_preds.extend(pred_class.tolist())
        cvm_targets.extend(true_class.tolist())
        #计算一下像素距离
        #padding =50/np.sqrt((gt_x[0, 0] - gt_x[4, 0])**2 + (gt_y[0, 0] - gt_y[4, 0])**2)  #hand物理像素距离
        padding = 25.4/244  # hand物理像素距离
        #提取真实值结束
        distances = np.linalg.norm(pred - gt, axis=1)
        with open('zuobiao.txt', 'a') as f:
            f.write(' '.join([f'{d:.1f}' for d in distances]) + '\n')
        #直接argmax后性能计算
        for j in range(Config.point_num):
            error = np.sqrt((gt[j][0] - pred[j][0]) ** 2 + (gt[j][1] - pred[j][1]) ** 2)
            loss[j] += error
            if error <= 1/padding:
                num_err_below_20[j] += 1
            elif error <= 2/padding:
                num_err_below_25[j] += 1
            elif error <= 2.5/padding:
                num_err_below_30[j] += 1
            elif error <= 3/padding:
                num_err_below_40[j] += 1
        #采样后的性能计算
        for jj in range(Config.point_num):
            error1 = np.sqrt((gt[jj][0] - zuobiao[jj][0]) ** 2 + (gt[jj][1] - zuobiao[jj][1]) ** 2)
            #error_list.append(f"{error1:.2f}")
            loss1[jj] += error1
            if error1 <= 1/padding:
                num1_err_below_20[jj] += 1
            elif error1 <= 2/padding:
                num1_err_below_25[jj] += 1
            elif error1 <= 2.5/padding:
                num1_err_below_30[jj] += 1
            elif error1 <= 3/padding:
                num1_err_below_40[jj] += 1
        # with open('error_list.txt','a') as f:
        #     f.write(' '.join(error_list) + '\n')
    loss = loss / (img_num + 1)
    num_err_below_25 = num_err_below_25 + num_err_below_20
    num_err_below_30 = num_err_below_30 + num_err_below_25
    num_err_below_40 = num_err_below_40 + num_err_below_30
    xiaoyu20 = sum(num_err_below_20/(img_num + 1))/Config.point_num
    xiaoyu25  = sum(num_err_below_25/(img_num + 1))/Config.point_num
    xiaoyu30 = sum(num_err_below_30 / (img_num + 1)) / Config.point_num
    xiaoyu40 = sum(num_err_below_40 / (img_num + 1)) / Config.point_num
    MRE = sum(loss)/Config.point_num

    #计算一下四个分类指标
    acc = accuracy_score(cvm_targets, cvm_preds)
    # prec = precision_score(cvm_targets, cvm_preds, average='macro', zero_division=0)
    # rec = recall_score(cvm_targets, cvm_preds, average='macro', zero_division=0)
    # f1 = f1_score(cvm_targets, cvm_preds, average='macro', zero_division=0)
    # print(f"分类结果：Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1 Score={f1:.4f}")
    print(classification_report(cvm_targets, cvm_preds, digits=4,zero_division=0))
    cm = confusion_matrix(cvm_targets, cvm_preds, labels=[0, 1, 2, 3, 4, 5])
    print(cm)
    #print(xiaoyu20)
    print(f"{xiaoyu20:.4f},{xiaoyu25:.4f},{xiaoyu30:.4f},{xiaoyu40:.4f},{MRE:.4f}")
    with open("预测.txt", "w", encoding="utf-8") as f:
        for img, values in results_dict.items():
            line = f"{img} {' '.join(values)}\n"
            f.write(line)
    if acc > best_acc:
        best_acc = acc
        best_xiaoyu20 = xiaoyu20
        torch.save(model.state_dict(), save_model_path)
        print(f"新最佳值 {best_xiaoyu20:.4f}")
        row0 = ['NO', '<=1', '<=2', '<=2.5', '<=3', 'mean_err']
        save_path = "output.txt"  # 保存文本文件的路径
        with open(save_path, 'w') as f:
            # 写入表头
            f.write('\t'.join(row0) + '\n')

            # 写入数据行
            for i in range(0, Config.point_num):
                line = [str(i + 1),
                        f"{num_err_below_20[i] / (img_num + 1):.4f}",
                        f"{num_err_below_25[i] / (img_num + 1):.4f}",
                        f"{num_err_below_30[i] / (img_num + 1):.4f}",
                        f"{num_err_below_40[i] / (img_num + 1):.4f}",
                        f"{loss[i]:.4f}"]
                f.write('\t'.join(line) + '\n')
            f.write('\t'.join([f"{xiaoyu20:.4f}", f"{xiaoyu25:.4f}", f"{xiaoyu30:.4f}", f"{xiaoyu40:.4f}", f"{MRE:.4f}"]) + '\n')

            # f.write('\t'.join([f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1 Score={f1:.4f}"]) + '\n')


    loss1 = loss1 / (img_num + 1)
    num1_err_below_25 = num1_err_below_25 + num1_err_below_20
    num1_err_below_30 = num1_err_below_30 + num1_err_below_25
    num1_err_below_40 = num1_err_below_40 + num1_err_below_30
    xiaoyu201 = sum(num1_err_below_20 / (img_num + 1)) / Config.point_num
    xiaoyu251 = sum(num1_err_below_25 / (img_num + 1)) / Config.point_num
    xiaoyu301 = sum(num1_err_below_30 / (img_num + 1)) / Config.point_num
    xiaoyu401 = sum(num1_err_below_40 / (img_num + 1)) / Config.point_num
    MRE1 =  sum(loss1)/Config.point_num
    # print(xiaoyu20)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"采样后{xiaoyu201:.4f},{xiaoyu251:.4f},{xiaoyu301:.4f},{xiaoyu401:.4f},{MRE1:.4f},用时{elapsed_time:.4f}")
    # if xiaoyu201 > best_xiaoyu201:
    #
    #     best_xiaoyu201 = xiaoyu201
    #     torch.save(model.state_dict(), save_model_path1)
    #     print(f"采样后最佳")
    #     spread_norm = (spread - spread.min(dim=1, keepdim=True)[0]) / (spread.max(dim=1, keepdim=True)[0] - spread.min(dim=1, keepdim=True)[0] + 1e-8)
    #     spread = F.softmax(spread_norm,dim=1)
    #     row0 = ['NO', '<=1', '<=2', '<=2.5', '<=3', 'mean_err','spread']
    #     save_path1 = "output1.txt"  # 保存文本文件的路径
    #     with open(save_path1, 'w') as f:
    #         # 写入表头
    #         f.write('\t'.join(row0) + '\n')
    #
    #         # 写入数据行
    #         for i in range(0, Config.point_num):
    #             line = [str(i + 1),
    #                     f"{num1_err_below_20[i] / (img_num + 1):.4f}",
    #                     f"{num1_err_below_25[i] / (img_num + 1):.4f}",
    #                     f"{num1_err_below_30[i] / (img_num + 1):.4f}",
    #                     f"{num1_err_below_40[i] / (img_num + 1):.4f}",
    #                     f"{loss1[i]:.4f}",
    #                     f"{spread[0][i][0].item():.4f}"
    #                     ]
    #             f.write('\t'.join(line) + '\n')
    #         f.write('\t'.join([f"{xiaoyu201:.4f}", f"{xiaoyu251:.4f}", f"{xiaoyu301:.4f}", f"{xiaoyu401:.4f}", f"{MRE1:.4f}"]) + '\n')


    #save_path1 = "out.txt"
    with open("out.txt","a") as f:
        f.write('\t'.join([f"{xiaoyu201:.4f}", f"{xiaoyu251:.4f}", f"{xiaoyu301:.4f}", f"{xiaoyu401:.4f}"]) + '\n')


# def predict(model, img_path):
#     img = Image.open(img_path)
#     img_w, img_h = img.size  # 注意这里是 (宽度, 高度)
#     transform = transforms.Compose([
#         transforms.Resize((Config.resize_h, Config.resize_w)),  # 调整大小
#         transforms.ToTensor(),  # 转换为tensor，已经将其转为 (C, H, W) 格式
#     ])
#
#     img_data = transform(img)
#     if img_data.shape[0] == 1:
#         img_data = img_data.repeat(3, 1, 1)
#     img_data = img_data.cuda(Config.GPU)
#     img_data = img_data.unsqueeze(0)
#     _,_,_,_,refined_coords,_,_,_,_,_= model(img_data)
#
#     zuobiao = refined_coords.cpu().detach().numpy()
#     zuobiao[:, :, 0] = zuobiao[:, :, 0] * (img_w - 1)
#     zuobiao[:, :, 1] = zuobiao[:, :, 1] * (img_h - 1)
#     zuobiao = zuobiao[0]
#     return zuobiao
