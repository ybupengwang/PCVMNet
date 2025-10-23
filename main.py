import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split,WeightedRandomSampler
from data import adaptive_sigma
from config import Config
from data import medical_dataset
from model import Farnet
from test import get_errors
from train import train_model
from samplingArgmax import SamplingArgmax
from loss import JointsOHKMMSELoss
from soft_argmax import SoftArgmax
from utils import setup_seed,worker_init_fn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import MultiStepLR
from collections import Counter
torch.cuda.empty_cache()
if __name__ == '__main__':
    setup_seed(25)
    model = Farnet()
    model.cuda(Config.GPU)

    # hrnet_path = "HR/pose_hrnet_w48_384x288.pth"
    # state_dict = torch.load(hrnet_path, map_location=torch.device(Config.GPU))
    # hrnet_dict = model.backbone.state_dict()
    # filtered_state_dict = {k: v for k, v in state_dict.items() if k in hrnet_dict and v.shape == hrnet_dict[k].shape}
    # hrnet_dict.update(filtered_state_dict)
    # vit导入
    vit_weights = torch.load('vit/vit_base_patch16_224_in21k.pth',
                             map_location=torch.device(Config.GPU))  # 通常是个state_dict字典
    model_vit_dict = model.vit_branch.state_dict()
    filtered_weights = {
        k: v for k, v in vit_weights.items()
        if k in model_vit_dict and v.shape == model_vit_dict[k].shape  # 看你实际写的名字
    }
    model_vit_dict.update(filtered_weights)
    model.vit_branch.load_state_dict(model_vit_dict)
    # vit导入结束
#定位导入
    model_weights_path = "best_model_cvm2dibiao.pth"  # 替换成你的权重路径best_moe_cvm2dibiao.pth
    state_dict = torch.load(model_weights_path, map_location=torch.device(Config.GPU))
    model_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape and "vit_branch" not in k}  #and "vit_branch" not in k
    model_dict.update(filtered_state_dict)
    model.load_state_dict(model_dict)

    #resnet导入
    # resnet_weights_path="resnet/resnet50-19c8e357.pth"
    # weights_dict = torch.load(resnet_weights_path, map_location="cpu")
    # for k in list(weights_dict.keys()):
    #     if "fc" in k:
    #         del weights_dict[k]
    # model.resnet_branch.load_state_dict(weights_dict, strict=False)
    # model.resnet_branch.cuda(Config.GPU)
    #resnet结束
    #swin 导入
    # swin_weights = "swin/swin_tiny_patch4_window7_224.pth"
    # weights_dict = torch.load(swin_weights, map_location=torch.device(Config.GPU))["model"]
    # for k in list(weights_dict.keys()):
    #     if "head" in k:
    #         del weights_dict[k]
    # print(model.swin_branch.load_state_dict(weights_dict, strict=False))
    #swin 导入结束

    for param in model.parameters():
        param.requires_grad = True
    for name, param in model.named_parameters():
        if "resnet_branch.bn" in name or "resnet_branch.layer4" in name or "resnet_branch.fc" in name:
            param.requires_grad = True
        if "vit_branch" in name or "vit_branch.cls_token" in name or "vit_branch.pos_embed" in name or "vit_branch.pos_embed" in name or "vit_branch.pre_logits.fc" in name or "vit_branch.prompt" in name or "vit_branch.patch_embed" in name or "vit_branch.landmark_embed" in name:
            param.requires_grad = True
        if "vit_branch.blocks" in name:
            param.requires_grad = False
    #
    # print([(name, param.requires_grad) for name, param in model.named_parameters()])



    soft_argmax = SoftArgmax()

    # 原先的 数据集
    test_set1 = medical_dataset(Config.test_img_dir1, Config.test_gt_dir1, Config.resize_h, Config.resize_w,Config.point_num, Config.sigma, transform=False)
    test_loader = DataLoader(dataset=test_set1, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,prefetch_factor=2,persistent_workers=True)
    train_set = medical_dataset(Config.img_dir, Config.gt_dir, Config.resize_h, Config.resize_w, Config.point_num,sigma=Config.sigma, transform=True)
    #过采样
    # labels = [int(train_set[i][11]) for i in range(len(train_set))]
    # class_counts = Counter(labels)
    # class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    # sample_weights = [class_weights[label] for label in labels]
    # weights = torch.DoubleTensor(sample_weights)
    # sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    #过采样
    #train_loader = DataLoader(dataset=train_set, batch_size=5, sampler=sampler,num_workers=4,pin_memory=True,prefetch_factor=2,persistent_workers=True)
    train_loader = DataLoader(dataset=train_set, batch_size=10,shuffle=True,num_workers=4, worker_init_fn=worker_init_fn,pin_memory=True, prefetch_factor=2, persistent_workers=True)
    #所有的随机分8:2
    # full_dataset = medical_dataset(Config.img_dir, Config.gt_dir, Config.resize_h, Config.resize_w, Config.point_num,sigma=Config.sigma, transform=True)
    # train_size = int(0.8 * len(full_dataset))
    # test_size = len(full_dataset) - train_size
    # generator = torch.Generator().manual_seed(42)
    # train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=5, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,prefetch_factor=2, persistent_workers=True)
    #所有的随机分8:2结束

    criterion = JointsOHKMMSELoss(use_target_weight=False).cuda(Config.GPU)

    #optimizer_ft = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-5)

    if Config.optimizer == 'adamw':
        optimizer_ft = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-5)
    elif Config.optimizer == 'adam':
        optimizer_ft = torch.optim.Adam(model.parameters(), lr=Config.lr)
    elif Config.optimizer == 'sgd':
        optimizer_ft = torch.optim.SGD(model.parameters(), lr=Config.lr, momentum=0.9, weight_decay=1e-4)
    elif Config.optimizer == 'rmsprop':
        optimizer_ft = torch.optim.RMSprop(model.parameters(), lr=Config.lr, alpha=0.99)
    elif Config.optimizer == 'adagrad':
        optimizer_ft = torch.optim.Adagrad(model.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_ft, lr_lambda=lambda epoch: (1 - epoch / Config.num_epochs) ** 3)
    #scheduler = CosineAnnealingLR(optimizer_ft, T_max=Config.num_epochs)#余弦退火
    #scheduler = MultiStepLR(optimizer_ft, milestones=[40, 80], gamma=0.5)
    for epoch in range(Config.num_epochs):
        if epoch%300 ==0:
            sigma = adaptive_sigma(epoch)
        #print("sigma："+str(sigma))
        for param_group in optimizer_ft.param_groups:
            print(f"sigma： {sigma}, Learning Rate: {param_group['lr']}")
        train_model(model, soft_argmax,criterion, optimizer_ft, scheduler, train_loader, test_loader, Config.num_epochs, epoch)
        #torch.save(model_ft, Config.save_model_path)

    #get_errors(model, test_loader, Config.test_gt_dir1, Config.save_results_path)
