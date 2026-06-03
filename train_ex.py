import time
import os
import sys

# 添加自定义路径到sys.path
# sys.path.append('/root/data/vjuicefs_ai_camera_jgroup_livephoto/11182563/color_transfer/methodDNCM/models')
from tqdm import tqdm
import copy
from options.ex_options import ExOptions  # 需自定义训练参数类
from data.data_loader import CreateDataLoader
from models.models import create_ex_model
import torchvision.utils as vutils
from util import util
from torch.optim import lr_scheduler
import torch
torch.cuda.empty_cache()
def batch_psnr(img1, img2, data_range=1.0):
    """
    计算4D张量的PSNR（支持批量计算）
    输入格式：[N, C, H, W]
    data_range: 1.0（归一化数据）或255（0-255范围）
    """
    mse = torch.mean((img1 - img2)**2, dim=(1,2,3))  # 计算每张图的MSE
    psnr = 10 * torch.log10(data_range**2 / mse)     # 计算每张图的PSNR
    return torch.mean(psnr)                           # 返回平均PSNR


if __name__ == '__main__':
    # 参数解析与初始化
    opt = ExOptions().parse()
    # opt.dataroot = '/root/data/vjuicefs_ai_camera_jgroup_livephoto/11182563/color_transfer/data_crop'
    opt.isTrain = True  # 标记训练模式
    # print(opt.dataroot)
    # torch.manual_seed(opt.seed)  # 设置随机种子[1,3](@ref)

    # 数据加载器配置
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print(f'Training images count: {dataset_size}')

    # 模型初始化
    model = create_ex_model(opt)
    model.net_train()  # 设置为训练模式[7](@ref)

    # 优化器配置（根据模型内部定义或外部定义）
    # if hasattr(model, 'optimizers'):
    #     optimizer = model.optimizers  # 假设模型已内置优化器 [5](@ref)
    # else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)


    #加载
    if opt.continue_train==True:
        optimizer_dict, scheduler_dict, epoch = model.net_train()  # 设置为训练模式[7](@ref)

        optimizer.load_state_dict(optimizer_dict)
        scheduler.load_state_dict(scheduler_dict)

        for param_group in optimizer.param_groups:
            #param_group['lr'] = 0.0001
            print(param_group['lr'])
        # print(scheduler.step_size)
        # scheduler.step_size = 50
    else:
        epoch=0

    # 训练循环配置
    total_steps = 0
    loss_history = []
    visual_dir = os.path.join(opt.checkpoints_dir, 'training_visuals')
    util.mkdirs([opt.checkpoints_dir, visual_dir])

    # 混合精度训练配置[8](@ref)
    # scaler = torch.cuda.amp.GradScaler(enabled=opt.fp16)

    # 训练主循环
    # for epoch in range(opt.epoch_count, opt.epochs):
    for epoch in range(epoch, opt.epochs):
        # continue
        epoch_start_time = time.time()
        epoch_iter = 0

        pbar = tqdm(dataset,
                    total=len(dataset)/opt.batchSize,
                    desc=f'Epoch {epoch}/{opt.epochs}',
                    dynamic_ncols=True)
        psnr_list=[1]
        net_list=[]
        for i, data in enumerate(pbar):
            # continue
            # if i==1000:
            #     break
            optimizer.zero_grad()

            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # 数据加载与模型输入
            model.set_input(data)  # 假设模型实现此方法[5](@ref)

            # 前向传播与损失计算
            # with torch.cuda.amp.autocast(enabled=opt.fp16):
            model.forward()
            losses = model.get_current_losses()  # 获取各损失项[5](@ref)

            if i%int(len(dataset)/opt.batchSize/12)==0 or i==int(len(dataset)/opt.batchSize-1):
                visuals = model.get_current_visuals()
                o1=visuals['img1_A']
                o2=visuals['out']
                psnr=batch_psnr(o1,o2,1.0)
                # net_list.append(model.clone())
                if psnr>min(psnr_list) and epoch % 2==0:
                    # print("save")
                    model.save_networks(epoch,optimizer,scheduler)  # 需实现网络保存方法[5](@ref)
                psnr_list.append(psnr)
                print(i, f'PSNR: {psnr} --Loss: {losses}')

            # 反向传播与优化
            losses.backward()
            optimizer.step()
            
            # scaler.scale(losses['total']).backward()  # 混合精度梯度缩放[8](@ref)
            # scaler.step(optimizer)
            # scaler.update()

            # 损失记录


        # max_value = max(psnr_list)  # 找到最大值
        # max_index = psnr_list.index(max_value)
        # print(max_index, psnr_list)
        scheduler.step()

            # 生成可视化结果
        if epoch % opt.display_freq == 0 :
            visuals = model.get_current_visuals()
            for key, val in visuals.items():
                # print(key,val)
                # if key.split('_')[-1] == 'output':
                
                vutils.save_image(val[0], '{}/{}_{}.png'.format(opt.display_dir, epoch, key), nrow=1, padding=0,
                                    normalize=False)

            # 学习率衰减
        # if 0 == epoch % opt.lr_decay_iters:
        #     scheduler.step()
            # model.update_learning_rate()  # 需在模型中实现[5](@ref)

        # 模型保存
        # if epoch % opt.save_epoch_freq == 0:
        #     model.save_networks(epoch,optimizer,scheduler)  # 需实现网络保存方法[5](@ref)
        visuals = model.get_current_visuals()
        # print(visuals['00_output1'])
        o1=visuals['img1_A']
        o2=visuals['out']
        # print(o1)
        print(f'Epoch {epoch} training time: {time.time()-epoch_start_time:.1f}s, Loss = {losses}, PSNR: {batch_psnr(o1,o2,1.0)}')

    # 最终模型保存
    model.save_networks('latest')
