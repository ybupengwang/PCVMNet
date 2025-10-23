# import os
# import numpy as np
#
# # 文件夹路径
# folder1 = r"D:\dataset\cepha400\AnnotationsByMD\400_senior"  # 医生1的标注文件夹
# folder2 = r"D:\dataset\cepha400\AnnotationsByMD\400_junior"  # 医生2的标注文件夹
# output_folder = r"D:\dataset\cepha400\AnnotationsByMD\400_adv"  # 存放平均结果的文件夹
#
# # 确保输出文件夹存在
# os.makedirs(output_folder, exist_ok=True)
#
# # 获取所有文件名（假设两个文件夹中的文件名完全一致）
# file_names = os.listdir(folder1)
#
# for file_name in file_names:
#     file_path1 = os.path.join(folder1, file_name)
#     file_path2 = os.path.join(folder2, file_name)
#     output_path = os.path.join(output_folder, file_name)
#
#     # 读取两个医生的标注数据
#     with open(file_path1, 'r') as f1, open(file_path2, 'r') as f2:
#         lines1 = f1.readlines()
#         lines2 = f2.readlines()
#
#     # 确保两个文件的行数相同
#     assert len(lines1) == len(lines2), f"File {file_name} has different line counts in two folders!"
#
#     averaged_data = []
#     for l1, l2 in zip(lines1, lines2):
#         l1 = l1.strip()
#         l2 = l2.strip()
#
#         # 处理坐标数据（两列数值，用逗号分割）
#         if "," in l1 and "," in l2:
#             x1, y1 = map(float, l1.split(","))
#             x2, y2 = map(float, l2.split(","))
#             avg_x, avg_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
#             averaged_data.append(f"{avg_x},{avg_y}")
#
#         # 处理等级数据（单个数值）
#         else:
#             v1 = float(l1)
#             v2 = float(l2)
#             avg_v = int((v1 + v2) / 2)
#             averaged_data.append(f"{avg_v}")
#
#     # 将平均后的数据写入新文件
#     with open(output_path, 'w') as out_f:
#         out_f.write("\n".join(averaged_data))
#
# print("所有文件处理完成，平均数据已保存至:", output_folder)

# import torch
# import torch.nn.functional as F
#
#
# def gumbel_softmax_sample(logits, tau=0.5):
#     """ 使用 Gumbel-Softmax 进行可微分采样 """
#     # Gumbel噪声，用于生成随机样本
#     gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
#     return F.softmax((logits + gumbel_noise) / tau, dim=-1)
#
#
# def differentiable_sampling(logits, W, H, tau=0.5):
#     """ 使用 Gumbel-Softmax 进行可微分采样，替代硬性采样 """
#     B, C, H, W = logits.shape
#     # 生成坐标网格（与logits相同设备）
#     y_grid, x_grid = torch.meshgrid(
#         torch.arange(H, device=logits.device),
#         torch.arange(W, device=logits.device),
#         indexing='ij'
#     )
#     coordinates = torch.stack([x_grid, y_grid], dim=-1).float()  # (H, W, 2)
#     coordinates = coordinates.view(-1, 2)  # (H*W, 2)
#     # 生成概率分布 (B, C, H*W)
#     prob = gumbel_softmax_sample(logits.view(B, C, -1), tau)  # 可导操作
#     # 计算期望坐标：概率加权平均（可导）
#     pred_coords = torch.matmul(prob, coordinates)  # (B, C, 2)
#     pred_coords[..., 0] /= (W - 1)  # x 归一化到 [0,1]
#     pred_coords[..., 1] /= (H - 1)  # y 归一化到 [0,1]
#     # 返回坐标
#     return pred_coords
#
#
# # ====== 示例 =======
# if __name__ == "__main__":
#     # 假设模型输出的预测热图 (logits)
#     logits = torch.randn(4, 1, 64, 64, requires_grad=True)  # (B, C, H, W)
#
#     # 进行可微分采样，获得预测坐标
#     pred_coords = differentiable_sampling(logits, 64, 64)
#
#     # 假设真实坐标
#     true_coords = torch.randint(0, 64, (4, 1, 2)).float()  # 真实坐标 (B, C, 2)
#
#     # 计算损失，例如 MSE 损失
#     loss = F.mse_loss(pred_coords, true_coords)
#
#     # 反向传播
#     loss.backward()
#
#     # 输出结果
#     print("预测坐标:", pred_coords)
#     print("梯度是否正确计算:", logits.grad is not None)


# import os
# import shutil
#
# def collect_images_by_person(root_dir, output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     # 遍历年月日文件夹，直到人名文件夹
#     for root, dirs, files in os.walk(root_dir):
#         for person in dirs:
#             person_path = os.path.join(root, person)
#             if os.path.isdir(person_path):
#                 # 检查是否为最后一层（即该文件夹下是图像而不是更多子目录）
#                 inner_items = os.listdir(person_path)
#                 if inner_items and all(os.path.isfile(os.path.join(person_path, item)) for item in inner_items):
#                     # 创建输出目录中的对应人名文件夹
#                     out_person_dir = os.path.join(output_dir, person)
#                     os.makedirs(out_person_dir, exist_ok=True)
#
#                     # 拷贝所有图片
#                     for file_name in inner_items:
#                         src_file = os.path.join(person_path, file_name)
#                         dst_file = os.path.join(out_person_dir, file_name)
#
#                         # 如果重名可以加前缀或跳过
#                         if os.path.exists(dst_file):
#                             base, ext = os.path.splitext(file_name)
#                             count = 1
#                             while os.path.exists(dst_file):
#                                 new_name = f"{base}_{count}{ext}"
#                                 dst_file = os.path.join(out_person_dir, new_name)
#                                 count += 1
#
#                         shutil.copy2(src_file, dst_file)
#
# # 用法示例
# root_dir = R'D:\dataset\眼科\2025'  # 比如 'data'
# output_dir = R'D:\dataset\眼科\2025\OUT'  # 比如 'collected_by_person'
# collect_images_by_person(root_dir, output_dir)

# import numpy as np
#
# def transform_preds_no_bbox(coords, image_size, output_size, use_udp=False):
#     """
#     将 heatmap 坐标恢复到原图坐标（整图作为输入，无边界框）。
#
#     Args:
#         coords (np.ndarray): (K, 2) 网络预测的关键点坐标（heatmap空间）
#         image_size (tuple): 原始图像尺寸 (width, height)
#         output_size (tuple): heatmap 尺寸 (width, height)
#         use_udp (bool): 是否使用无偏 UDP 方法
#
#     Returns:
#         np.ndarray: 恢复后的原图坐标 (K, 2)
#     """
#     assert coords.shape[1] == 2, "仅支持 (K, 2) 的关键点坐标输入"
#     W, H = image_size
#     W_hm, H_hm = output_size
#     coords = coords.copy()
#
#     if use_udp:
#         # 构造 scale/center（仿照UDP标准）
#         scale = np.array([W / 200.0, H / 200.0], dtype=np.float32)
#         center = np.array([W / 2.0, H / 2.0], dtype=np.float32)
#
#         scale_x = scale[0] * 200.0 / (W_hm - 1.0)
#         scale_y = scale[1] * 200.0 / (H_hm - 1.0)
#
#         target_coords = np.zeros_like(coords)
#         target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 200.0 * 0.5
#         target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 200.0 * 0.5
#     else:
#         # 普通比例映射（非UDP）
#         scale_x = W / W_hm
#         scale_y = H / H_hm
#         target_coords = np.zeros_like(coords)
#         target_coords[:, 0] = coords[:, 0] * scale_x
#         target_coords[:, 1] = coords[:, 1] * scale_y
#
#     return target_coords
#
# heatmap_coords = np.array([[32, 48], [16, 16], [60, 10]])  # 假设是64x64 heatmap预测出的关键点
# image_size = (256, 256)  # 原图尺寸
# output_size = (64, 64)   # heatmap大小
#
# # 调用
# orig_coords = transform_preds_no_bbox(heatmap_coords, image_size, output_size, use_udp=True)
# print(orig_coords)

# import numpy as np
# from typing import Tuple, Optional
# import torch
# import torch.nn as nn
# from scipy.ndimage import gaussian_filter
# from itertools import product
# def generate_udp_gaussian_heatmaps(
#     heatmap_size: Tuple[int, int],
#     keypoints: np.ndarray,
#     keypoints_visible: np.ndarray,
#     sigma: float,
# ) -> Tuple[np.ndarray]:
#     """Generate gaussian heatmaps of keypoints using `UDP`_.
#
#     Args:
#         heatmap_size (Tuple[int, int]): Heatmap size in [W, H]
#         keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
#         keypoints_visible (np.ndarray): Keypoint visibilities in shape
#             (N, K)
#         sigma (float): The sigma value of the Gaussian heatmap
#
#     Returns:
#         tuple:
#         - heatmaps (np.ndarray): The generated heatmap in shape
#             (K, H, W) where [W, H] is the `heatmap_size`
#         - keypoint_weights (np.ndarray): The target weights in shape
#             (N, K)
#
#     .. _`UDP`: https://arxiv.org/abs/1911.07524
#     """
#
#     N, K, _ = keypoints.shape
#     W, H = heatmap_size
#
#     heatmaps = np.zeros((K, H, W), dtype=np.float32)
#     keypoint_weights = keypoints_visible.copy()
#
#     # 3-sigma rule
#     radius = sigma * 3
#
#     # xy grid
#     gaussian_size = 2 * radius + 1
#     x = np.arange(0, gaussian_size, 1, dtype=np.float32)
#     y = x[:, None]
#
#     for n, k in product(range(N), range(K)):
#         # skip unlabled keypoints
#         if keypoints_visible[n, k] < 0.5:
#             continue
#
#         mu = (keypoints[n, k] + 0.5).astype(np.int64)
#         # check that the gaussian has in-bounds part
#         left, top = (mu - radius).astype(np.int64)
#         right, bottom = (mu + radius + 1).astype(np.int64)
#
#         if left >= W or top >= H or right < 0 or bottom < 0:
#             keypoint_weights[n, k] = 0
#             continue
#
#         mu_ac = keypoints[n, k]
#         x0 = y0 = gaussian_size // 2
#         x0 += mu_ac[0] - mu[0]
#         y0 += mu_ac[1] - mu[1]
#         gaussian = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
#
#         # valid range in gaussian
#         g_x1 = max(0, -left)
#         g_x2 = min(W, right) - left
#         g_y1 = max(0, -top)
#         g_y2 = min(H, bottom) - top
#
#         # valid range in heatmap
#         h_x1 = max(0, left)
#         h_x2 = min(W, right)
#         h_y1 = max(0, top)
#         h_y2 = min(H, bottom)
#
#         heatmap_region = heatmaps[k, h_y1:h_y2, h_x1:h_x2]
#         gaussian_regsion = gaussian[g_y1:g_y2, g_x1:g_x2]
#
#         _ = np.maximum(heatmap_region, gaussian_regsion, out=heatmap_region)
#
#     return heatmaps
# def get_heatmap_maximum(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """Get maximum response location and value from heatmaps.
#
#     Note:
#         batch_size: B
#         num_keypoints: K
#         heatmap height: H
#         heatmap width: W
#
#     Args:
#         heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)
#
#     Returns:
#         tuple:
#         - locs (np.ndarray): locations of maximum heatmap responses in shape
#             (K, 2) or (B, K, 2)
#         - vals (np.ndarray): values of maximum heatmap responses in shape
#             (K,) or (B, K)
#     """
#     assert isinstance(heatmaps,
#                       np.ndarray), ('heatmaps should be numpy.ndarray')
#     assert heatmaps.ndim == 3 or heatmaps.ndim == 4, (
#         f'Invalid shape {heatmaps.shape}')
#
#     if heatmaps.ndim == 3:
#         K, H, W = heatmaps.shape
#         B = None
#         heatmaps_flatten = heatmaps.reshape(K, -1)
#     else:
#         B, K, H, W = heatmaps.shape
#         heatmaps_flatten = heatmaps.reshape(B * K, -1)
#
#     y_locs, x_locs = np.unravel_index(
#         np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
#     locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
#     vals = np.amax(heatmaps_flatten, axis=1)
#     locs[vals <= 0.] = -1
#
#     if B:
#         locs = locs.reshape(B, K, 2)
#         vals = vals.reshape(B, K)
#
#     return locs, vals
# def gaussian_blur2(hm, kernel):
#     sigma = kernel / 6  # 经验公式：kernel=6*sigma 可近似匹配 OpenCV 效果
#     border = (kernel - 1) // 2
#
#     batch_size, num_joints, height, width = hm.shape
#
#     for i in range(batch_size):
#         for j in range(num_joints):
#             origin_max = np.max(hm[i, j])
#             # 创建带边界的扩展热图
#             dr = np.zeros((height + 2 * border, width + 2 * border), dtype=np.float32)
#             dr[border: -border, border: -border] = hm[i, j].copy()
#             # 使用 scipy 进行高斯模糊
#             dr = gaussian_filter(dr, sigma=sigma)
#             # 裁剪回原始大小
#             cropped = dr[border: -border, border: -border]
#             if np.max(cropped) > 0:
#                 hm[i, j] = cropped * (origin_max / np.max(cropped))
#             else:
#                 hm[i, j] = cropped
#     return hm
# def gaussian_blur1(hm, kernel):
#     sigma = kernel / 6  # 经验公式：kernel=6*sigma 可近似匹配 OpenCV 效果
#     border = (kernel - 1) // 2
#
#     num_joints, height, width = hm.shape
#
#     for j in range(num_joints):
#         origin_max = np.max(hm[j])
#         # 创建带边界的扩展热图
#         dr = np.zeros((height + 2 * border, width + 2 * border), dtype=np.float32)
#         dr[border: -border, border: -border] = hm[j].copy()
#         # 使用 scipy 进行高斯模糊
#         dr = gaussian_filter(dr, sigma=sigma)
#         # 裁剪回原始大小
#         cropped = dr[border: -border, border: -border]
#         if np.max(cropped) > 0:
#             hm[j] = cropped * (origin_max / np.max(cropped))
#         else:
#             hm[j] = cropped
#     return hm
# def refine_keypoints_dark_udp(keypoints: np.ndarray, heatmaps: np.ndarray,
#                               blur_kernel_size: int) -> np.ndarray:
#     """Refine keypoint predictions using distribution aware coordinate decoding
#     for UDP. See `UDP`_ for details. The operation is in-place.
#
#     Note:
#
#         - instance number: N
#         - keypoint number: K
#         - keypoint dimension: D
#         - heatmap size: [W, H]
#
#     Args:
#         keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
#         heatmaps (np.ndarray): The heatmaps in shape (K, H, W)
#         blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
#             modulation
#
#     Returns:
#         np.ndarray: Refine keypoint coordinates in shape (N, K, D)
#
#     .. _`UDP`: https://arxiv.org/abs/1911.07524
#     """
#     N, K = keypoints.shape[:2]
#     H, W = heatmaps.shape[1:]
#
#     # modulate heatmaps
#     heatmaps = gaussian_blur1(heatmaps, blur_kernel_size)
#     np.clip(heatmaps, 1e-3, 50., heatmaps)
#     np.log(heatmaps, heatmaps)
#
#     heatmaps_pad = np.pad(
#         heatmaps, ((0, 0), (1, 1), (1, 1)), mode='edge').flatten()
#
#     for n in range(N):
#         index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
#         index += (W + 2) * (H + 2) * np.arange(0, K)
#         index = index.astype(int).reshape(-1, 1)
#         i_ = heatmaps_pad[index]
#         ix1 = heatmaps_pad[index + 1]
#         iy1 = heatmaps_pad[index + W + 2]
#         ix1y1 = heatmaps_pad[index + W + 3]
#         ix1_y1_ = heatmaps_pad[index - W - 3]
#         ix1_ = heatmaps_pad[index - 1]
#         iy1_ = heatmaps_pad[index - 2 - W]
#
#         dx = 0.5 * (ix1 - ix1_)
#         dy = 0.5 * (iy1 - iy1_)
#         derivative = np.concatenate([dx, dy], axis=1)
#         derivative = derivative.reshape(K, 2, 1)
#
#         dxx = ix1 - 2 * i_ + ix1_
#         dyy = iy1 - 2 * i_ + iy1_
#         dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
#         hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
#         hessian = hessian.reshape(K, 2, 2)
#         hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
#         keypoints[n] -= np.einsum('imn,ink->imk', hessian,
#                                   derivative).squeeze()
#
#     return keypoints
#
# class UDPHeatmap(nn.Module):
#     r"""Generate keypoint heatmaps by Unbiased Data Processing (UDP).
#     See the paper: `The Devil is in the Details: Delving into Unbiased Data
#     Processing for Human Pose Estimation`_ by Huang et al (2020) for details.
#
#     Note:
#
#         - instance number: N
#         - keypoint number: K
#         - keypoint dimension: D
#         - image size: [w, h]
#         - heatmap size: [W, H]
#
#     Encoded:
#
#         - heatmap (np.ndarray): The generated heatmap in shape (C_out, H, W)
#             where [W, H] is the `heatmap_size`, and the C_out is the output
#             channel number which depends on the `heatmap_type`. If
#             `heatmap_type=='gaussian'`, C_out equals to keypoint number K;
#             if `heatmap_type=='combined'`, C_out equals to K*3
#             (x_offset, y_offset and class label)
#         - keypoint_weights (np.ndarray): The target weights in shape (K,)
#
#     Args:
#         input_size (tuple): Image size in [w, h]
#         heatmap_size (tuple): Heatmap size in [W, H]
#         heatmap_type (str): The heatmap type to encode the keypoitns. Options
#             are:
#
#             - ``'gaussian'``: Gaussian heatmap
#             - ``'combined'``: Combination of a binary label map and offset
#                 maps for X and Y axes.
#
#         sigma (float): The sigma value of the Gaussian heatmap when
#             ``heatmap_type=='gaussian'``. Defaults to 2.0
#         radius_factor (float): The radius factor of the binary label
#             map when ``heatmap_type=='combined'``. The positive region is
#             defined as the neighbor of the keypoit with the radius
#             :math:`r=radius_factor*max(W, H)`. Defaults to 0.0546875
#         blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
#             modulation in DarkPose. Defaults to 11
#
#     .. _`The Devil is in the Details: Delving into Unbiased Data Processing for
#     Human Pose Estimation`: https://arxiv.org/abs/1911.07524
#     """
#
#     def __init__(self,
#                  input_size: Tuple[int, int],
#                  heatmap_size: Tuple[int, int],
#                  sigma: float = 2.,
#                  blur_kernel_size: int = 11) -> None:
#         super().__init__()
#         self.input_size = input_size
#         self.heatmap_size = heatmap_size
#         self.sigma = sigma
#
#
#         self.blur_kernel_size = blur_kernel_size
#         self.scale_factor = ((np.array(input_size) - 1) /
#                              (np.array(heatmap_size) - 1)).astype(np.float32)
#
#
#
#     def encode(self,
#                keypoints: np.ndarray) -> np.ndarray:
#         """Encode keypoints into heatmaps. Note that the original keypoint
#         coordinates should be in the input image space.
#
#         Args:
#             keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
#             keypoints_visible (np.ndarray): Keypoint visibilities in shape
#                 (N, K)
#
#         Returns:
#             dict:
#             - heatmap (np.ndarray): The generated heatmap in shape
#                 (C_out, H, W) where [W, H] is the `heatmap_size`, and the
#                 C_out is the output channel number which depends on the
#                 `heatmap_type`. If `heatmap_type=='gaussian'`, C_out equals to
#                 keypoint number K; if `heatmap_type=='combined'`, C_out
#                 equals to K*3 (x_offset, y_offset and class label)
#             - keypoint_weights (np.ndarray): The target weights in shape
#                 (K,)
#         """
#         assert keypoints.shape[0] == 1, (
#             f'{self.__class__.__name__} only support single-instance '
#             'keypoint encoding')
#
#
#         keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)
#
#
#         heatmaps = generate_udp_gaussian_heatmaps(
#             heatmap_size=self.heatmap_size,
#             keypoints=keypoints / self.scale_factor,
#             keypoints_visible=keypoints_visible,
#             sigma=self.sigma)
#
#
#
#
#         return heatmaps
#
#     def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """Decode keypoint coordinates from heatmaps. The decoded keypoint
#         coordinates are in the input image space.
#
#         Args:
#             encoded (np.ndarray): Heatmaps in shape (K, H, W)
#
#         Returns:
#             tuple:
#             - keypoints (np.ndarray): Decoded keypoint coordinates in shape
#                 (N, K, D)
#             - scores (np.ndarray): The keypoint scores in shape (N, K). It
#                 usually represents the confidence of the keypoint prediction
#         """
#         heatmaps = encoded.copy()
#
#         keypoints, scores = get_heatmap_maximum(heatmaps)
#         # unsqueeze the instance dimension for single-instance results
#         keypoints = keypoints[None]
#         scores = scores[None]
#
#         keypoints = refine_keypoints_dark_udp(
#             keypoints, heatmaps, blur_kernel_size=self.blur_kernel_size)
#
#         W, H = self.heatmap_size
#         keypoints = keypoints / [W - 1, H - 1] * self.input_size
#         return keypoints, scores
#     def forward(self,x, mode="encode"):
#         """
#                 Forward method that can encode or decode based on mode.
#                 Args:
#                     x: keypoints or heatmaps
#                     mode: "encode" or "decode"
#                 """
#         if mode == "encode":
#
#             return self.encode(x)
#         elif mode == "decode":
#             return self.decode(x)
#
#
# udp = UDPHeatmap(input_size = (1935,2400),heatmap_size=(512, 512), sigma=2.0)
# keypoints = np.array([[[779, 1068], [1352, 983], [1234, 1274]]], dtype=np.float32)
# heatmaps = udp(keypoints, mode="encode")
# zuobiao = udp(heatmaps,mode="decode")

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
#
#
# class TransformerCoordRefiner(nn.Module):
#     def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2):
#         super().__init__()
#         self.input_fc = nn.Linear(input_dim, d_model)  # coords+cov+feat
#         encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
#         self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.output_fc = nn.Linear(d_model, 2)  # refined坐标输出 (x, y)
#
#     def forward(self, coords, cov, sampled_feat):
#         """
#         coords: (B, N, 2)
#         cov: (B, N, 2, 2)
#         sampled_feat: (B, N, C)
#         """
#         B, N, _ = coords.shape
#
#         cov_feat = cov.reshape(B, N, -1)  # (B, N, 4)
#         x = torch.cat([coords, cov_feat, sampled_feat], dim=-1)  # (B, N, D)
#         x = self.input_fc(x)              # (B, N, d_model)
#         x = x.permute(1, 0, 2)            # (N, B, d_model)
#         x = self.transformer(x)           # (N, B, d_model)
#         x = x.permute(1, 0, 2)            # (B, N, d_model)
#         refined_coords = self.output_fc(x)  # (B, N, 2)
#         return refined_coords


# B, N, C = 2, 5, 115
# coords = torch.rand(B, N, 2)                      # 模拟坐标 (0~1)
# cov = torch.rand(B, N, 2, 2) * 0.1                # 不确定性协方差矩阵
# sampled_feat = torch.randn(B, N, C)               # 从图中采样的特征
# input_dim = 2 + 4 + C  # coords (2) + cov (2x2) + feat
# model = TransformerCoordRefiner(input_dim=input_dim)
# refined_coords = model(coords, cov, sampled_feat)
# print("输入 coords:")
# print(coords)
# print("\n输出 refined_coords:")
# print(refined_coords)

# import os
# import shutil
# txt_folder = 'D:\dataset\yachi\肺标注\chest_labels\chest\labels'        # 你的txt文件夹路径
# png_folder = 'D:\dataset\yachi\肺标注\chest_labels\chest\image'        # 你的png文件夹路径
# output_folder = 'D:\dataset\yachi\肺标注\chest_labels\chest\img'  # 你想保存对应png的文件夹路径
#
# # 确保输出文件夹存在
# os.makedirs(output_folder, exist_ok=True)
#
# # 获取txt文件名（不带后缀）
# txt_filenames = [os.path.splitext(f)[0] for f in os.listdir(txt_folder) if f.endswith('.txt')]
#
# # 遍历txt文件名，查找对应png文件，复制到输出文件夹
# for name in txt_filenames:
#     png_path = os.path.join(png_folder, name + '.png')
#     if os.path.exists(png_path):
#         shutil.copy(png_path, output_folder)
#         print(f'Copied: {name}.png')
#     else:
#         print(f'Warning: {name}.png not found in {png_folder}')

# from PIL import Image
# import matplotlib.pyplot as plt
#
# #加载图像
# #image_path = r'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\jingzhuitestimg\111381401.jpg'  # chenren
# image_path = r'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\头测-an\第二次补充等级\21年缺失等级\11451430.jpg' #ertong
# image = Image.open(image_path).convert('L')

#假设地标点坐标（单位为像素）
# landmarks = [
#     (815, 1060),   # 地标1
#     (1390, 882),   # 地标2
#     (1308, 1178),  # 地标3
#     (603, 1218),   # 地标4
#     (1494, 1479),  # 地标5
#     (1426, 1781),  # 地标6
#     (1425, 1961),  # 地标7
#     (1377, 2020),  # 地标8
#     (1415, 2002),  # 地标9
#     (798, 1805),   # 地标10
#     (1509, 1594),  # 地标11
#     (1592, 1658),  # 地标12
#     (1675, 1523),  # 地标13
#     (1599, 1793),  # 地标14
#     (1602, 1416),  # 地标15
#     (1498, 1995),  # 地标16
#     (1016, 1439),  # 地标17
#     (1488, 1404),  # 地标18
#     (665, 1384),   # 地标19
# ]
# landmarks = [
#     (1257.6367, 1614.0286),
#     (1298.2303, 1618.1322),
#     (1346.4818, 1638.1077),
#     (1253.875, 1646.0535),
#     (1333.39, 1743.5621),
#     (1236.151, 1762.651),
#     (1296.5885, 1772.7081),
#     (1331.325, 1787.7263),
#     (1240.8958, 1781.5706),
#     (1324.1364, 1800.7053),
#     (1229.2322, 1898.2059),
#     (1259.4999, 1901.8655),
#     (1274.559, 1910.0541)
# ]
#儿童
# landmarks = [
#     (1294, 1537),
#     (1346, 1545),
#     (1410, 1572),
#     (1289, 1585),
#     (1394, 1640),
#     (1270, 1672),
#     (1326, 1687),
#     (1389, 1712),
#     (1259, 1720),
#     (1356, 1780),
#     (1229, 1791),
#     (1280, 1820),
#     (1339, 1850)
# ]
#成人
# landmarks = [
#     (1269, 1470),
#     (1329, 1473),
#     (1367, 1501),
#     (1258, 1516),
#     (1348, 1537),
#     (1250, 1630),
#     (1300, 1630),
#     (1355, 1649),
#     (1241, 1680),
#     (1350, 1699),
#     (1246, 1794),
#     (1298, 1787),
#     (1362, 1801)
# ]
# origin = (0, 0)
#
# # 创建图像
# # plt.figure(figsize=(8, 6))
# plt.imshow(image, cmap='gray')
# plt.scatter(*zip(*landmarks), c='red', s=2)     # 地标点
# # plt.scatter(*origin, c='blue', s=60, marker='x', label='Origin')    # 原点
#
# # 添加编号标注
# # for idx, (x, y) in enumerate(landmarks, start=1):
# #     plt.text(x + 5, y - 5, f'{idx}', color='orange', fontsize=12, weight='bold')
#
# plt.title("Medical Landmark Annotation")
# plt.axis('off')
# # plt.legend()
# plt.tight_layout()
# #plt.savefig(r'D:\ShareCache\王朋 (0000008400)\0NK\科研整理\cvm\ertong.pdf', format='pdf', bbox_inches='tight')
# plt.show()


# image_path = r'D:\dataset\yachi\shou-biaozhu\images\5332.jpg'  # 替换为你的图像路径
# image = Image.open(image_path).convert('L')
#
# # 假设地标点坐标（单位为像素）
# landmarks = [
#     (579, 2248),   # 地标1
#     (604, 2148),   # 地标2
#     (740, 2273),   # 地标3
#     (753, 2248),   # 地标4
#     (1071, 2273),  # 地标5
#     (1066, 2097),  # 地标6
#     (685, 2105),   # 地标7
#     (806, 2169),   # 地标8
#     (930, 2156),   # 地标9
#     (775, 2036),   # 地标10
#     (879, 2061),   # 地标11
#     (1034, 1935),  # 地标12
#     (1083, 1995),  # 地标13
#     (664, 1855),   # 地标14
#     (742, 1812),   # 地标15
#     (864, 1797),   # 地标16
#     (996, 1802),   # 地标17
#     (1183, 1902),  # 地标18
#     (1473, 1544),  # 地标19
#     (1617, 1254),  # 地标20
#     (1724, 1067),  # 地标21
#     (1088, 1079),  # 地标22
#     (1171, 675),   # 地标23
#     (1215, 443),   # 地标24
#     (1227, 260),   # 地标25
#     (844, 1061),   # 地标26
#     (845, 593),    # 地标27
#     (840, 295),    # 地标28
#     (818, 100),    # 地标29
#     (638, 1181),   # 地标30
#     (557, 745),    # 地标31
#     (496, 463),    # 地标32
#     (482, 256),    # 地标33
#     (454, 1320),   # 地标34
#     (319, 1016),   # 地标35
#     (233, 833),    # 地标36
#     (187, 653),    # 地标37
# ]
#
# # 原点（左上角为几何原点）
# origin = (0, 0)
#
# # 创建图像
# # plt.figure(figsize=(8, 6))
# plt.imshow(image, cmap='gray')
# plt.scatter(*zip(*landmarks), c='red', s=10)     # 地标点
# # plt.scatter(*origin, c='blue', s=60, marker='x', label='Origin')    # 原点
#
# # 添加编号标注
# for idx, (x, y) in enumerate(landmarks, start=1):
#     plt.text(x + 5, y - 5, f'{idx}', color='orange', fontsize=8, weight='bold')
#
# plt.title("Medical Landmark Annotation")
# plt.axis('off')
# # plt.legend()
# plt.tight_layout()
# plt.savefig('landmarks_annotation.pdf', format='pdf', bbox_inches='tight')
# plt.show()


# import os
# from PIL import Image
#
# # 文件夹路径（请替换为你的真实路径）
# image_folder = "D:\dataset\yachi\肺标注\chest_labels\chest\img"              # 图片文件夹
# landmark_folder = "D:\dataset\yachi\肺标注\chest_labels\chest\labels"    # 存放归一化坐标的txt文件夹
# output_folder = "D:\dataset\yachi\肺标注\chest_labels\chest\zhenshilabels"         # 输出真实坐标的txt文件夹
#
# # 创建输出目录
# os.makedirs(output_folder, exist_ok=True)
#
# # 遍历所有图片
# for filename in os.listdir(image_folder):
#     if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
#         name, _ = os.path.splitext(filename)
#         img_path = os.path.join(image_folder, filename)
#         txt_path = os.path.join(landmark_folder, f"{name}.txt")
#         out_txt_path = os.path.join(output_folder, f"{name}.txt")
#
#         # 获取图像尺寸
#         try:
#             with Image.open(img_path) as img:
#                 w, h = img.size
#         except Exception as e:
#             print(f"❌ Cannot open image {filename}: {e}")
#             continue
#
#         # 读取归一化坐标
#         try:
#             with open(txt_path, 'r') as f:
#                 lines = f.readlines()
#         except FileNotFoundError:
#             print(f"⚠️ Landmark file not found for image {filename}")
#             continue
#
#         num_points = int(lines[0].strip())
#         coords = [list(map(float, line.strip().split())) for line in lines[1:num_points+1]]
#
#         # 反归一化
#         real_coords = [[round(x * (w-1), 2), round(y * (h-1), 2)] for x, y in coords]
#
#         # 保存新坐标
#         with open(out_txt_path, 'w') as f:
#
#             for x, y in real_coords:
#                 f.write(f"{x} {y}\n")
#
# print("✅ 所有地标点已成功转换为真实像素坐标。")




# ###从json中提取txt坐标
# import os
# import json
# # 设置你的json文件所在目录
# json_dir = r"D:\dataset\yachi\tianjinkouqing\第三批\21年\21年统计\头颅"  # 修改为你自己的目录路径
# for filename in os.listdir(json_dir):
#     if filename.endswith(".json"):
#         json_path = os.path.join(json_dir, filename)
#         txt_filename = os.path.splitext(filename)[0] + ".txt"
#         txt_path = os.path.join(json_dir, txt_filename)
#
#         with open(json_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#
#         # 提取点并按 label 整理为字典
#         label_point_map = {}
#         for shape in data.get("shapes", []):
#             if shape.get("shape_type") == "point":
#                 label = shape.get("label")
#                 if label.isdigit():
#                     x, y = shape["points"][0]
#                     x_int = round(x)
#                     y_int = round(y)
#                     label_point_map[int(label)] = f"{x_int},{y_int}"
#
#         # 根据 label 升序排序，并写入对应行
#         sorted_lines = []
#         for label in sorted(label_point_map.keys()):
#             sorted_lines.append(label_point_map[label])
#         if len(sorted_lines)!=13:
#             print({txt_filename})
#             continue  # 跳过这个文件，不写入
#         # 写入txt文件，每行一个点
#         with open(txt_path, 'w', encoding='utf-8') as f:
#             f.write("\n".join(sorted_lines))

        #print(f"已处理: {txt_filename}")





#文件夹 和子文件名字提取

# import os
#
# def get_all_chains(base_path):
#     chains = []
#
#     def walk_chain(current_path, current_chain):
#         subdirs = [d for d in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, d))]
#         if not subdirs:
#             chains.append(current_chain)
#         else:
#             for sub in subdirs:
#                 walk_chain(os.path.join(current_path, sub), current_chain + [sub])
#
#     # 从第一层子目录开始
#     for item in os.listdir(base_path):
#         full_path = os.path.join(base_path, item)
#         if os.path.isdir(full_path):
#             walk_chain(full_path, [item])
#
#     return chains
#
# def write_all_subfolder_chains(base_path, output_path):
#     chains = get_all_chains(base_path)
#     with open(output_path, 'w', encoding='utf-8') as f:
#         for chain in chains:
#             f.write(" ".join(chain) + "\n")
#
# # 修改成你的文件夹路径
# base_folder = r"D:\dataset\yachi\tianjinkouqing\第三批\21年\21年\21年全"  # 例如：r"C:\Users\yourname\Desktop\test"
# output_txt = r"D:\dataset\yachi\tianjinkouqing\第三批\21年\21年\21年全\output.txt"
# write_all_subfolder_chains(base_folder, output_txt)


####################修改名字变成文件夹名字 ##########################
# import os
# import re
# def extract_id(folder_name):
#     # 提取编号部分（假设是以数字开头）
#     match = re.match(r'^(\d+)', folder_name)
#     return match.group(1) if match else None
#
# def rename_images_with_id(base_path):
#     rename_counter = {}  # 用于编号的计数
#
#     for root, dirs, files in os.walk(base_path):
#         parent_folder = os.path.basename(root)
#         folder_id = extract_id(parent_folder)
#         if not folder_id:
#             continue  # 跳过无法识别编号的文件夹
#
#         jpg_files = [f for f in files if f.lower().endswith('.jpg')]
#         if not jpg_files:
#             continue
#
#         if folder_id not in rename_counter:
#             rename_counter[folder_id] = 0
#
#         for file in jpg_files:
#             old_path = os.path.join(root, file)
#
#             count = rename_counter[folder_id]
#             new_name = f"{folder_id}.jpg" if count == 0 else f"{folder_id}_{count}.jpg"
#             rename_counter[folder_id] += 1
#
#             new_path = os.path.join(root, new_name)
#
#             if os.path.exists(new_path):
#                 print(f"⚠️ 文件已存在，跳过: {new_path}")
#                 continue
#
#             os.rename(old_path, new_path)
#             print(f"✅ 重命名: {old_path} → {new_path}")
#
# # 用法
# base_folder = r"D:\dataset\yachi\tianjinkouqing\第三批\21年\21年\21第二部分"
# rename_images_with_id(base_folder)


# import os
# import shutil
#
# def copy_all_jpgs_keep_name(src_dir, dst_dir):
#     if not os.path.exists(dst_dir):
#         os.makedirs(dst_dir)
#
#     name_count = {}
#
#     for root, dirs, files in os.walk(src_dir):
#         for file in files:
#             if file.lower().endswith(".jpg"):
#                 src_file = os.path.join(root, file)
#
#                 # 保留原始文件名
#                 base_name = file
#                 name, ext = os.path.splitext(base_name)
#
#                 # 如果文件名重复，自动加编号
#                 if base_name in name_count:
#                     count = name_count[base_name]
#                     new_name = f"{name}_{count}{ext}"
#                     name_count[base_name] += 1
#                 else:
#                     new_name = base_name
#                     name_count[base_name] = 1
#
#                 dst_file = os.path.join(dst_dir, new_name)
#
#                 shutil.copy2(src_file, dst_file)
#                 print(f"✅ 已复制: {src_file} → {dst_file}")
#
# # ✅ 替换为你的路径
# source_folder = r"D:\dataset\yachi\tianjinkouqing\第三批\21年\21年\21年全"
# target_folder = r"D:\dataset\yachi\tianjinkouqing\第三批\21年\21年统计\曲面"
#
# copy_all_jpgs_keep_name(source_folder, target_folder)

# import os
# import csv
# import shutil
#
# # 设置路径
# csv_path = r"D:\dataset\眼科\眼前节图像\1.裂隙灯数据集\标注信息\annotations-2024-12-22-13.csv"                 # CSV 文件路径
# image_root = r"D:\dataset\眼科\眼前节图像\1.裂隙灯数据集\图像数据"     # 所有图片所在文件夹路径
# output_root = r"D:\dataset\眼科\眼前节图像\有标签数据"             # 输出文件夹根路径
#
# # 标签名（按列顺序，和 CSV 对应）
# labels = [
#     "外伤",
#     "病变：上睑下垂睑内翻",
#     "病变：其他",
#     "病变：白内障",
#     "病变：肿物",
#     "病变：胬肉",
#     "病变：角膜炎",
#     "病变：青光眼"
# ]
#
# # 创建每个标签对应的输出文件夹
# for label in labels:
#     os.makedirs(os.path.join(output_root, label), exist_ok=True)
#
# # 读取 CSV 文件
# with open(csv_path, 'r', encoding='utf-8-sig') as f:
#     reader = csv.reader(f)
#     headers = next(reader)  # 跳过表头
#     for row in reader:
#         filename = row[0]
#         label_flags = row[1:]
#
#         src_path = os.path.join(image_root, filename)
#         if not os.path.exists(src_path):
#             print(f"⚠️ 图像未找到：{filename}")
#             continue
#
#         for idx, flag in enumerate(label_flags):
#             if flag == '1':
#                 label_name = labels[idx]
#                 dst_path = os.path.join(output_root, label_name, filename)
#
#                 # 复制图片
#                 shutil.copy2(src_path, dst_path)
#                 print(f"✅ 已复制 {filename} → {label_name}")



#################随机分配训练集和测试集
# import os
# import random
#
# source_dir = r'D:\dataset\眼科\眼前节图像\有标签数据'  # 原始数据路径
# train_txt = r'D:\dataset\眼科\眼前节图像\有标签数据\train.txt'
# test_txt = r'D:\dataset\眼科\眼前节图像\有标签数据\test.txt'
# train_ratio = 0.7
#
# # 类别列表，用于编码标签
# categories = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])
# category_to_label = {cat: idx for idx, cat in enumerate(categories)}
# print(category_to_label)
#
# train_lines = []
# test_lines = []
#
# for category in categories:
#     category_path = os.path.join(source_dir, category)
#     images = [f for f in os.listdir(category_path)
#               if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]
#     random.shuffle(images)
#
#     split_idx = int(len(images) * train_ratio)
#     train_imgs = images[:split_idx]
#     test_imgs = images[split_idx:]
#
#     for img in train_imgs:
#         line = f"{os.path.join(category_path, img)} {category_to_label[category]}"
#         train_lines.append(line)
#
#     for img in test_imgs:
#         line = f"{os.path.join(category_path, img)} {category_to_label[category]}"
#         test_lines.append(line)
#
# # 写入txt文件
# with open(train_txt, 'w', encoding='utf-8') as f:
#     f.write('\n'.join(train_lines))
#
# with open(test_txt, 'w', encoding='utf-8') as f:
#     f.write('\n'.join(test_lines))
#
# print(f"写入完成：训练集 {len(train_lines)}，测试集 {len(test_lines)}")


# import os
# import json
#
# # 配置路径
# txt_path = r"D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\头测-an\wenjianming.txt"  # 包含编号.jpg 的 txt 文件路径
# json_dir = r"D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\头测-an\24-2头颅"      # json 文件夹路径
# # 读取所有原始行
# with open(txt_path, 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#
# updated_lines = []
# for line in lines:
#     line = line.strip()
#     if not line.endswith('.jpg'):
#         updated_lines.append(line)
#         continue
#
#     base_name = os.path.splitext(line)[0]
#     json_path = os.path.join(json_dir, base_name + ".json")
#
#     description_to_add = ""
#     if os.path.exists(json_path):
#         with open(json_path, 'r', encoding='utf-8') as jf:
#             print(json_path)
#             data = json.load(jf)
#             for shape in data.get("shapes", []):
#                 desc = shape.get("description", "")
#                 if isinstance(desc, str) and desc.strip():  # 找到非空description就跳出
#                     description_to_add = desc.strip()
#                     break
#
#     if description_to_add:
#         updated_lines.append(f"{line} {description_to_add}")
#     else:
#         updated_lines.append(line)
#
# # 覆盖写回原txt文件
# with open(txt_path, 'w', encoding='utf-8') as f:
#     f.write("\n".join(updated_lines))
#
# print("✅ 已完成，原始 txt 文件已更新。")

# import os
# import shutil
# def append_level_to_txt(summary_txt_path, txt_folder, all_files_folder, missing_folder):
#     os.makedirs(missing_folder, exist_ok=True)
#
#     with open(summary_txt_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 2:
#                 filename, level = parts
#             elif len(parts) == 1:
#                 filename = parts[0]
#                 level = None
#             else:
#                 continue  # 跳过格式不对的行
#
#             base_name = os.path.splitext(filename)[0]  # 去掉 .jpg
#             txt_path = os.path.join(txt_folder, f"{base_name}.txt")
#
#             if os.path.exists(txt_path):
#                 if level is not None:
#                     with open(txt_path, 'rb+') as t:
#                         t.seek(-1, os.SEEK_END)
#                         last_char = t.read(1)
#                         if last_char != b'\n':
#                             t.write(b'\n')  # 补换行
#                         t.write(f"{level}\n".encode())
#                 else:
#                     print(f"⚠️ 跳过 {filename}，无等级信息，移动相关文件中...")
#
#                     for ext in ['.jpg', '.json', '.txt']:
#                         src = os.path.join(all_files_folder, base_name + ext)
#                         dst = os.path.join(missing_folder, base_name + ext)
#                         if os.path.exists(src):
#                             shutil.move(src, dst)
#                             print(f"  ✅ 已移动：{src} → {dst}")
#                         else:
#                             print(f"  ⚠️ 文件不存在：{src}")
#
#             else:
#                 print(f"❌ 找不到对应 txt 文件：{txt_path}")
#
# summary_txt_path = "D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\头测-an\wenjianming.txt"
#
# # 所有单个编号.txt 文件所在的目录
# txt_folder = r"D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\头测-an\21-1头颅/"
# missing_folder = r'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\头测-an\21-1头颅/no_level_files/'
# all_files_folder = txt_folder
#
# append_level_to_txt(summary_txt_path, txt_folder, all_files_folder, missing_folder)


# import os
# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk
#
#
# class ImageTxtEditor:
#     def __init__(self, root, folder):
#         self.root = root
#         self.root.title("图像+TXT 标注查看器")
#         self.folder = folder
#         self.files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg'))]
#         self.index = 0
#
#         self.image_label = tk.Label(root)
#         self.image_label.pack()
#
#         self.text_box = tk.Text(root, height=3, width=60)
#         self.text_box.pack()
#
#         self.button_frame = tk.Frame(root)
#         self.button_frame.pack()
#
#         tk.Button(self.button_frame, text="上一张", command=self.prev_image).pack(side=tk.LEFT)
#         tk.Button(self.button_frame, text="保存修改", command=self.save_txt).pack(side=tk.LEFT)
#         tk.Button(self.button_frame, text="下一张", command=self.next_image).pack(side=tk.LEFT)
#
#         self.load_image()
#
#     def load_image(self):
#         if self.index >= len(self.files):
#             return
#
#         filename = self.files[self.index]
#         img_path = os.path.join(self.folder, filename)
#         txt_path = os.path.join(self.folder, os.path.splitext(filename)[0] + ".txt")
#
#         # 加载图片
#         img = Image.open(img_path).resize((702, 534))
#         self.photo = ImageTk.PhotoImage(img)
#         self.image_label.config(image=self.photo)
#
#         # 确保可编辑
#         self.text_box.config(state=tk.NORMAL)
#         self.text_box.delete("1.0", tk.END)
#
#         if os.path.exists(txt_path):
#             with open(txt_path, 'r', encoding='utf-8') as f:
#                 lines = f.readlines()
#             if len(lines) >= 14:
#                 self.text_box.insert(tk.END, lines[13].strip())
#
#         # 设置光标位置并聚焦
#         self.text_box.mark_set(tk.INSERT, "1.0")
#         self.text_box.focus_set()
#         self.root.lift()
#         self.root.attributes('-topmost', True)
#         self.root.after(100, lambda: self.root.attributes('-topmost', False))
#
#     def save_txt(self):
#         filename = self.files[self.index]
#         txt_path = os.path.join(self.folder, os.path.splitext(filename)[0] + ".txt")
#
#         new_line = self.text_box.get("1.0", tk.END).strip() + "\n"
#
#         if os.path.exists(txt_path):
#             with open(txt_path, 'r', encoding='utf-8') as f:
#                 lines = f.readlines()
#         else:
#             lines = []
#         # 补齐到14行
#         while len(lines) < 13:
#             lines.append("\n")
#         if len(lines) == 13:
#             lines.append("\n")
#             lines.append(new_line)
#         else:
#             lines[13] = new_line
#
#         with open(txt_path, 'w', encoding='utf-8') as f:
#             f.writelines(lines)
#
#     def next_image(self):
#         self.save_txt()
#         if self.index < len(self.files) - 1:
#             self.index += 1
#             self.load_image()
#
#     def prev_image(self):
#         self.save_txt()
#         if self.index > 0:
#             self.index -= 1
#             self.load_image()
#
# if __name__ == "__main__":
#     root = tk.Tk()
#     folder = filedialog.askdirectory(title="请选择图像+txt文件夹")
#     if folder:
#         app = ImageTxtEditor(root, folder)
#         root.mainloop()
#
#
# import os
# import shutil
#
# img_folder = r'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\antrainimg'
# label_folder = r'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\antrainlabel'
# save_img_folder = r'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\img_aug'
# save_label_folder = r'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\label_aug'
#
# os.makedirs(save_img_folder, exist_ok=True)
# os.makedirs(save_label_folder, exist_ok=True)
#
# # 支持的图片扩展名
# img_exts = ['.jpg', '.png', '.jpeg', '.bmp']
#
# for filename in os.listdir(label_folder):
#     if filename.endswith('.txt'):
#         label_path = os.path.join(label_folder, filename)
#
#         with open(label_path, 'r') as f:
#             lines = f.readlines()
#
#         if len(lines) >= 14 and lines[13].strip() == 'cs6':
#             base_name = os.path.splitext(filename)[0]
#
#             # 找到对应的图像文件
#             img_path = None
#             for ext in img_exts:
#                 candidate_path = os.path.join(img_folder, base_name + ext)
#                 if os.path.exists(candidate_path):
#                     img_path = candidate_path
#                     break
#
#             if img_path is None:
#                 print(f"未找到图像: {base_name}")
#                 continue
#
#             # 复制图像和标签10次
#             for i in range(5, 7):
#                 new_name = f'copy_{i}_{base_name}'
#                 new_img_name = new_name + os.path.splitext(img_path)[1]
#                 new_label_name = new_name + '.txt'
#
#                 shutil.copy(img_path, os.path.join(save_img_folder, new_img_name))
#                 shutil.copy(label_path, os.path.join(save_label_folder, new_label_name))
#
# print("复制完成。")

#
# import os
#
# # 设置你要遍历的文件夹路径
# folder_path = r'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\头测-an\第二次补充等级\2324输出文件夹路径'  # 例如：r'C:\images'
#
# # 设置输出txt文件名
# output_txt = r'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\头测-an\第二次补充等级\2324输出文件夹路径\image_list.txt'
#
# # 支持的图像扩展名
# image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
#
# # 打开txt文件准备写入
# with open(output_txt, 'w') as f:
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if os.path.splitext(file)[1].lower() in image_extensions:
#                 f.write(file + '\n')  # 只写图像文件名（不包含路径）


# import os
# import tkinter as tk
# from tkinter import filedialog, messagebox
# from PIL import Image, ImageTk
#
# class ImageTxtNavigator:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("天津市眼科医院标注")
#         self.root.geometry("820x950")
#
#         # ✅ 创建可滚动框架容器
#         main_frame = tk.Frame(self.root)
#         main_frame.pack(fill="both", expand=True)
#
#         canvas = tk.Canvas(main_frame)
#         scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
#         self.scrollable_frame = tk.Frame(canvas)
#
#         self.scrollable_frame.bind(
#             "<Configure>",
#             lambda e: canvas.configure(
#                 scrollregion=canvas.bbox("all")
#             )
#         )
#
#         canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
#         canvas.configure(yscrollcommand=scrollbar.set)
#
#         canvas.pack(side="left", fill="both", expand=True)
#         scrollbar.pack(side="right", fill="y")
#
#         # 以下控件全部改为放在 scrollable_frame 中 ✅
#         self.progress_label = tk.Label(self.scrollable_frame, text="当前: 0 / 0", font=("Arial", 14))
#         self.progress_label.pack(pady=5)
#
#         self.canvas = tk.Canvas(self.scrollable_frame, width=702, height=534, bg='gray')
#         self.canvas.pack(pady=10)
#
#         self.label_options = ["上睑下垂睑内翻", "其他", "肿物","白内障", "胬肉", "青光眼", "角膜溃疡", "角膜炎", "角膜白斑","角膜穿孔","翼状胬肉","外伤","结膜肿物"]
#         self.check_vars = {}
#         check_frame = tk.Frame(self.scrollable_frame)
#         check_frame.pack(pady=5)
#         tk.Label(check_frame, text="选择标签（多选框）:", font=("Arial", 12)).grid(row=0, column=0, columnspan=4, sticky='w')
#
#         for idx, label in enumerate(self.label_options):
#             var = tk.BooleanVar(value=False)
#             cb = tk.Checkbutton(check_frame, text=label, variable=var, font=("Arial", 12))
#             row = idx // 4 + 1
#             col = idx % 4
#             cb.grid(row=row, column=col, sticky='w', padx=10, pady=3)
#             self.check_vars[label] = var
#
#         prob_frame = tk.Frame(self.scrollable_frame)
#         prob_frame.pack(pady=5)
#         tk.Label(prob_frame, text="模型置信度Top-2:", font=("Arial", 12)).pack(side=tk.LEFT)
#         self.prob_display_var = tk.StringVar()
#         self.prob_display_label = tk.Label(prob_frame, textvariable=self.prob_display_var, font=("Arial", 12), width=20, relief='sunken', anchor='center')
#         self.prob_display_label.pack(side=tk.LEFT, padx=5)
#
#         self.text_var = tk.StringVar()
#         self.entry = tk.Entry(self.scrollable_frame, textvariable=self.text_var, font=("Arial", 18), width=40, justify='center')
#         self.entry.pack(pady=10)
#
#         self.button_frame = tk.Frame(self.scrollable_frame)
#         self.button_frame.pack(pady=10)
#
#         self.load_folder_btn = tk.Button(self.button_frame, text="选择文件夹", command=self.load_folder)
#         self.load_folder_btn.pack(side=tk.LEFT, padx=10)
#
#         self.prev_btn = tk.Button(self.button_frame, text="上一张", command=self.prev_image)
#         self.prev_btn.pack(side=tk.LEFT, padx=10)
#
#         self.next_btn = tk.Button(self.button_frame, text="下一张Enter切换", command=self.next_image)
#         self.next_btn.pack(side=tk.LEFT, padx=10)
#
#         self.save_btn = tk.Button(self.button_frame, text="保存当前", command=self.save_txt)
#         self.save_btn.pack(side=tk.LEFT, padx=10)
#
#         # 不变部分
#         self.image_paths = []
#         self.current_index = -1
#         self.current_txt_path = None
#
#         self.root.bind("<Return>", lambda event: self.next_image())
#         self.root.bind("<Left>", lambda event: self.prev_image())
#         self.root.bind("<Right>", lambda event: self.next_image())
#
#     def update_progress_label(self):
#         total = len(self.image_paths)
#         current = self.current_index + 1 if total > 0 else 0
#         if total > 0 and 0 <= self.current_index < total:
#             filename = os.path.basename(self.image_paths[self.current_index])
#         else:
#             filename = "无"
#         self.progress_label.config(text=f"当前: {current} / {total}    文件名: {filename}")
#
#     def save_last_index(self):
#         with open("last_index.txt", "w",encoding='utf-8') as f:
#             f.write(str(self.current_index))
#
#     def load_last_index(self):
#         if os.path.exists("last_index.txt"):
#             try:
#                 with open("last_index.txt", "r",encoding='utf-8') as f:
#                     idx = int(f.read().strip())
#                     return idx
#             except:
#                 return -1
#         return -1
#
#     def load_folder(self):
#         folder = filedialog.askdirectory()
#         if not folder:
#             return
#
#         self.image_paths = [
#             os.path.join(folder, f)
#             for f in os.listdir(folder)
#             if f.lower().endswith((".jpg", ".jpeg", ".png"))
#         ]
#         self.image_paths.sort()
#
#         last_idx = self.load_last_index()
#         if 0 <= last_idx < len(self.image_paths):
#             self.current_index = last_idx
#         else:
#             self.current_index = 0
#
#         self.show_image()
#
#     def next_image(self):
#         if not self.image_paths:
#             return
#
#         self.save_txt()
#
#         if self.current_index < len(self.image_paths) - 1:
#             self.current_index += 1
#             self.show_image()
#         else:
#             messagebox.showinfo("提示", "已经是最后一张了！")
#
#     def prev_image(self):
#         if not self.image_paths:
#             return
#
#         self.save_txt()
#
#         if self.current_index > 0:
#             self.current_index -= 1
#             self.show_image()
#         else:
#             messagebox.showinfo("提示", "已经是第一张了！")
#
#     def show_image(self):
#         img_path = self.image_paths[self.current_index]
#         self.display_image(img_path)
#         self.load_txt(img_path)
#         self.save_last_index()
#         self.update_progress_label()
#
#     def display_image(self, img_path):
#         image = Image.open(img_path)
#         image.thumbnail((702, 534))
#         self.photo = ImageTk.PhotoImage(image)
#         self.canvas.delete("all")
#         self.canvas.create_image(351, 270, image=self.photo)
#
#     def load_txt(self, img_path):
#         txt_path = os.path.splitext(img_path)[0] + ".txt"
#         self.current_txt_path = txt_path
#
#         # 清空所有复选框和文本框
#         for var in self.check_vars.values():
#             var.set(False)
#         self.prob_display_var.set("")
#         self.text_var.set("")
#
#         if os.path.exists(txt_path):
#             with open(txt_path, "r",encoding='utf-8') as f:
#                 lines = [line.strip() for line in f if line.strip()]
#                 if not lines:
#                     return
#
#                 labels_probs = []
#                 text_lines = []
#                 label_only_lines = []
#
#                 for line in lines:
#                     parts = line.split()
#                     # 有概率值的标签
#                     if len(parts) == 2 and parts[0] in self.label_options:
#                         try:
#                             prob = float(parts[1])
#                             labels_probs.append((parts[0], prob))
#                         except:
#                             text_lines.append(line)
#                     # 没有概率值的标签
#                     elif line in self.label_options:
#                         label_only_lines.append(line)
#                     else:
#                         text_lines.append(line)
#
#                 if labels_probs:
#                     # 情况1：只勾选最大概率标签
#                     # max_label, max_prob = max(labels_probs, key=lambda x: x[1])
#                     # self.check_vars[max_label].set(True)
#                     # self.prob_display_var.set(f"{max_label}  {max_prob:.3f}")
#
#                     # Top-2 标签按概率排序
#                     sorted_probs = sorted(labels_probs, key=lambda x: x[1], reverse=True)
#                     top1 = sorted_probs[0]
#                     self.check_vars[top1[0]].set(True)  # 只勾选最大标签
#
#                     # 显示Top-2标签
#                     top_display = [f"{label} {prob:.3f}" for label, prob in sorted_probs[:2]]
#                     self.prob_display_var.set(" / ".join(top_display))
#
#                 elif label_only_lines:
#                     # 情况2：没有概率，只勾选所有标签
#                     for label in label_only_lines:
#                         self.check_vars[label].set(True)
#
#                 # 显示剩余文本
#                 self.text_var.set("\n".join(text_lines))
#         else:
#             self.text_var.set("")
#
#         self.entry.focus()
#         self.entry.icursor(tk.END)
#
#     def save_txt(self):
#         if self.current_txt_path:
#             text = self.text_var.get().strip()
#             with open(self.current_txt_path, "w",encoding='utf-8') as f:
#                 # if text:
#                 #     # 输入框有内容，只保存输入框内容
#                 #     f.write(text + "\n")
#                 # else:
#                 #     # 输入框为空，保存所有选中的复选框标签，每行一个
#                 #     for label, var in self.check_vars.items():
#                 #         if var.get():
#                 #             f.write(label + "\n")
#                 for label, var in self.check_vars.items():
#                     if var.get():
#                         f.write(label + "\n")
#
#                 f.write(text + "\n")
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = ImageTxtNavigator(root)
#     root.mainloop()
# import os
# label_file_path = r"D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\头测-an\第二次补充等级\张阳标注\2324image_list.txt"  # 替换为实际路径
# base_dir = r"D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\头测-an\24-2头颅\no_level_files"
#
# with open(label_file_path, "r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()
#         if not line:
#             continue
#
#         parts = line.split()
#         if len(parts) != 2:
#             continue
#
#         img_name, label = parts
#         file_id = img_name.split(".")[0]
#         txt_filename = os.path.join(base_dir, f"{file_id}.txt")
#
#         # 如果目标文件不存在就跳过
#         if not os.path.exists(txt_filename):
#             continue
#
#         # 将标签转换为小写，例如 CS3 -> cs3
#         label = label.lower()
#
#         # 检查是否需要先添加换行
#         with open(txt_filename, "rb") as check_file:
#             check_file.seek(-1, os.SEEK_END)
#             last_char = check_file.read(1)
#             need_newline = last_char != b'\n'
#
#         # 以追加模式写入
#         with open(txt_filename, "a", encoding="utf-8") as txt_file:
#             if need_newline:
#                 txt_file.write("\n")
#             txt_file.write(label + "\n")
#
# import os
# from PIL import Image
#
# # 设置图像文件夹路径
# folder_path = r"D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\antestimg"  # 修改为你的路径
#
# # 目标尺寸
# target_size = (2808, 2136)
#
# # 遍历文件夹下所有图片文件
# for filename in os.listdir(folder_path):
#     if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
#         image_path = os.path.join(folder_path, filename)
#         try:
#             with Image.open(image_path) as img:
#                 if img.size != target_size:
#                     print(f"{filename} size is {img.size}")
#         except Exception as e:
#             print(f"Error reading {filename}: {e}")

# import matplotlib.pyplot as plt
# # 设置字体为 Times New Roman，字号为 14
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 14
# # 模拟数据
# vlp_counts = [1, 2, 3, 6, 12]
# acc = [66.11, 66.97, 68.06, 69.17, 72.50]
# precision = [52.44, 66.35, 71.01, 62.91, 65.51]
# recall = [53.42, 58.54, 59.30, 60.94, 64.52]
# f1 = [51.87, 56.84, 59.49, 61.36, 64.28]
#
# # 绘图
# plt.figure(figsize=(7.5, 4.5))
# plt.plot(vlp_counts, acc, marker='o', label='Accuracy')
# plt.plot(vlp_counts, precision, marker='s', label='Precision')
# plt.plot(vlp_counts, recall, marker='^', label='Recall')
# plt.plot(vlp_counts, f1, marker='d', label='F1-score')
#
# plt.xlabel('Number of VLP Blocks')
# plt.ylabel('Performance (%)')
# #plt.title('Effect of VLP Block Number on Model Performance')
#
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("block number.pdf", format="pdf", )
# plt.show()


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

# 固定横坐标
numbers = [1, 2, 3, 4, 5]

# 各项 MRE 数据
mre_avg =   [1.12, 1.11, 1.09, 1.10, 1.09]
mre_lm4 =   [1.83, 1.86, 1.80, 1.79, 1.86]
mre_lm10 =  [1.81, 1.69, 1.76, 1.60, 1.70]
mre_lm19 =  [1.52, 1.62, 1.49, 1.58, 1.61]

# 各项 SDR 数据
sdr_avg =   [86.91, 87.82, 88.39, 88.74, 88.53]
sdr_lm4 =   [63.00, 66.00, 68.67, 69.33, 66.00]
sdr_lm10 =  [64.67, 70.00, 66.00, 70.00, 68.00]
sdr_lm19 =  [74.00, 72.00, 76.00, 75.33, 73.33]

# 设置画布
plt.figure(figsize=(12, 5))

# ========== MRE 图 ==========
plt.subplot(1, 2, 1)
plt.plot(numbers, mre_avg, marker='o', label='Avg', linewidth=2)
plt.plot(numbers, mre_lm4, marker='s', label='Landmark4', linewidth=2)
plt.plot(numbers, mre_lm10, marker='^', label='Landmark10', linewidth=2)
plt.plot(numbers, mre_lm19, marker='d', label='Landmark19', linewidth=2)
# plt.title('MRE vs Number')
plt.xlabel('Number')
plt.ylabel('MRE (mm)')
plt.xticks(numbers)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right', bbox_to_anchor=(1.0, 0.55))

# ========== SDR 图 ==========
plt.subplot(1, 2, 2)
plt.plot(numbers, sdr_avg, marker='o', label='Avg', linewidth=2)
plt.plot(numbers, sdr_lm4, marker='s', label='Landmark4', linewidth=2)
plt.plot(numbers, sdr_lm10, marker='^', label='Landmark10', linewidth=2)
plt.plot(numbers, sdr_lm19, marker='d', label='Landmark19', linewidth=2)
# plt.title('SDR (2mm) vs Number')
plt.xlabel('Number')
plt.ylabel('SDR (%)')
plt.xticks(numbers)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right', bbox_to_anchor=(1.00, 0.9))

# 调整布局
plt.tight_layout()
plt.savefig("expert number.pdf", format="pdf", )
# 显示图像
plt.show()


