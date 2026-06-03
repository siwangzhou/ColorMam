import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from data.wavelet import IWT,DWT
from models.models import create_ex_model
import torchvision.utils as vutils
from util import util
from torchvision import transforms as T
import torch

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
    print(torch.cuda.is_available(),'----------------------------')
    print(torch.cuda.device_count(),'-----------------------------')
    opt = TestOptions().parse()
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_ex_model(opt)
    opt.is_psnr = True

    summary_dir = opt.display_dir
    util.mkdirs([summary_dir])

    dwt=DWT().cuda()
    iwt=IWT().cuda()

    for i, data in enumerate(dataset):
        # if i >= opt.how_many:
        #     break
        # model.set_input(data)
        # print(data['input'].cuda(),data['target'].cuda())
        # print(data['input'].shape)
        # dwt_images_i=dwt(data['input'].cuda())
        # a1=dwt_images_i
        # a1=torch.clip(a1,0,1)
        # a=T.ToPILImage()(a1[0])
        # a.show()
        # a = T.ToPILImage()(a1[1]*4)
        # a.show()
        # a = T.ToPILImage()(a1[2]*4)
        # a.show()

        # dwt_images_t = dwt(data['target'].cuda())
        # a2 = dwt_images_t
        # a = T.ToPILImage()(a)
        # a.show()
        # print(a1.shape)
        # print(batch_psnr(a1[0:1],a2[0:1]),batch_psnr(a1[1:2],a2[1:2]),batch_psnr(a1[2:3],a2[2:3]),batch_psnr(a1[3:4],a2[3:4]))
        # out = model.test(dwt_images_i[:1,:,:,:].cuda(), dwt_images_t[:1,:,:,:].cuda())
        # dwt_images_i[0]=out
        # out = model.test(dwt_images_i[1:2, :, :, :].cuda(), dwt_images_t[1:2, :, :, :].cuda())
        # dwt_images_i[1] = out
        # out = model.test(dwt_images_i[2:3, :, :, :].cuda(), dwt_images_t[2:3, :, :, :].cuda())
        # dwt_images_i[2] = out
        # out=iwt(dwt_images_i)
        t1=time.time()
        out=model.test(data['input'].cuda()*2-1, data['target'].cuda()*2-1)/2+0.5
        print(time.time()-t1,'s,========================')
        # torch.cuda.empty_cache()
        # continue
        print('\nimage: ', i, '/', len(dataset))

        # visuals = model.get_current_visuals()
        print('%04d: process image... ' % (i))
        # for key, val in visuals.items():
        # print(data['name'])
            # if key.split('_')[-1]=='output':
        vutils.save_image(out, '{}/{}.png'.format(summary_dir, data['name'][0].split('\\')[-1]), nrow=1, padding = 0, normalize = False)
