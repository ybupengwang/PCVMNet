import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from config import Config
from model import Farnet
import matplotlib.pyplot as plt
from utils import get_markposion_fromtxt
from data import get_final_preds
import os
import json
def load_model(model, checkpoint_path, device='cpu'):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)  # 严格对应
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model

def predict(model, img_path, device='cpu'):
    # 加载图像
    img = Image.open(img_path)

    img_w, img_h = img.size  # (宽, 高)

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((Config.resize_h, Config.resize_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(img)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img_data = img.unsqueeze(0).to(device)  # 添加 batch 维度，变成 (1, C, H, W)

    # 模型前向推理
    with torch.no_grad():
        outputs,_,_,_,refined_coords,log_sigma,_,_,_,_,cvm = model(img_data)
        # 这里根据你的模型输出调整解包逻辑
        # if isinstance(outputs, (list, tuple)) and len(outputs) >= 5:
        #     refined_coords = outputs[4]
        # else:
        #     raise ValueError("模型输出格式不正确，无法获取 refined_coords")

    # 坐标还原到原始图像尺寸
    # refined_coords = refined_coords.cpu().numpy()  # (B, N, 2)
    # refined_coords[:, :, 0] = refined_coords[:, :, 0] * (img_w - 1)
    # refined_coords[:, :, 1] = refined_coords[:, :, 1] * (img_h - 1)

    #利用热图后处理 泰勒展开
    preds, _ = get_final_preds(outputs.cpu().detach().numpy())
    preds[:, :, 0] = preds[:, :, 0] * (img_w - 1) / (Config.resize_w - 1)
    preds[:, :, 1] = preds[:, :, 1] * (img_h - 1) / (Config.resize_h - 1)
    pred = preds[0]  ##为了测试后处理方法效果

    return pred,cvm  # 返回第一个样本的坐标

def save_keypoints_to_json(keypoints, img_path, save_json_dir):
    # 读取图像获取尺寸
    image = Image.open(img_path)
    width, height = image.size

    # 获取图片名
    img_name = os.path.basename(img_path)

    # 构造 shapes 列表，按 label 从 1 开始
    shapes = []
    for idx, (x, y) in enumerate(keypoints, start=1):
        shape = {
            "label": str(idx),
            "points": [[float(round(x, 12)), float(round(y, 12))]],
            "group_id": None,
            "description": "",
            "shape_type": "point",
            "flags": {},
            "mask": None
        }
        shapes.append(shape)

    # 构造完整 JSON 数据
    json_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    # 保存 json 文件
    os.makedirs(save_json_dir, exist_ok=True)
    json_path = os.path.join(save_json_dir, os.path.splitext(img_name)[0] + ".json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"已保存 JSON 到: {json_path}")

# 模型初始化（你需要根据自己模型定义来构建）

if __name__ == '__main__':
    # model = Farnet()
    # checkpoint_path = "best_moe_cvm1.pth"
    # model = load_model(model, checkpoint_path, device='cpu')
    #
    # # 预测图像关键点
    # img_path = r"D:\dataset\yachi\tianjinkouqing\第三批\21年\21年统计\头颅\11442701.jpg"
    # note_gt_road =  'D:\dataset\yachi/tianjinkouqing/第三批/21年/21年统计\头颅/' + img_path.split('\\')[-1].split('.')[0] + '.txt'
    # gt_x, gt_y = get_markposion_fromtxt(Config.point_num, note_gt_road)
    # gt_x = np.trunc(np.reshape(gt_x, (Config.point_num, 1)))
    # gt_y = np.trunc(np.reshape(gt_y, (Config.point_num, 1)))
    # gt = np.concatenate((gt_x, gt_y), 1)
    #
    # image = Image.open(img_path)
    # keypoints = predict(model, img_path, device='cpu')
    # print(keypoints)  # (N, 2)，表示 N 个关键点
    # landmarks = np.array(keypoints)
    # plt.imshow(image, cmap='gray')
    # plt.scatter(gt[:, 0], gt[:, 1], c='green', s=15, label='Ground Truth')
    # plt.scatter(*landmarks.T, c='red', s=10)
    # for idx, (x, y) in enumerate(landmarks, start=1):
    #     plt.text(x + 5, y - 5, f'{idx}', color='orange', fontsize=12, weight='bold')
    #
    # plt.title("Medical Landmark Annotation")
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()


    #批量推理写入txt中，txt再转到json 把21年的数据完善一下
    model = Farnet()
    checkpoint_path = "best_moe_cvm2.pth"
    model = load_model(model, checkpoint_path, device='cpu')
    # 预测图像关键点
    img_dir = r"D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\antestimg"
    for filename in os.listdir(img_dir):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(img_dir, filename)
            try:
                # 打开图像并预测关键点
                keypoints,cvm = predict(model, img_path, device='cpu')
                pred_class = torch.argmax(cvm, dim=1).item()  # 结果是 int 类型，比如 2
                with open(r"D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\antestimg\predictions.txt", "a") as f:
                    f.write(f"{filename} {pred_class}\n")
                # 保存 keypoints 为 json
                # save_keypoints_to_json(keypoints, img_path, save_json_dir=img_dir)


            except Exception as e:
                print(f"❌ 错误: {filename} 处理失败: {e}")
    # keypoints = predict(model, img_path, device='cpu')
    # save_keypoints_to_json(keypoints, img_path,save_json_dir=r"D:\dataset\yachi\tianjinkouqing\第三批\21年\21年统计\头颅")
    # save_dir = r"D:\dataset\yachi\tianjinkouqing\第三批\21年\21年统计\头颅"
    # os.makedirs(save_dir, exist_ok=True)
    #
    # # 获取不带扩展名的文件名
    # basename = os.path.splitext(os.path.basename(img_path))[0]
    # txt_path = os.path.join(save_dir, basename + ".txt")
    #
    # # 写入关键点到 txt 文件
    # with open(txt_path, "w") as f:
    #     for (x, y) in keypoints:
    #         f.write(f"{x:.12f},{y:.12f}\n")
    #
    # print(f"关键点已保存到: {txt_path}")




