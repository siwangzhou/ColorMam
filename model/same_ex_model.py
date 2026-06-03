import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from .resnet_ex import SimpleResNet,color_fuse_net
from . import networks
from torchvision import transforms as T
import sys
#from torch.utils.serialization import load_lua
import torch.nn as nn
import torchvision
import random
import cv2
# import torchfile
import torch.nn.functional as F
from models.ARCHI.SaMam_model import SaMam
from torch.jit import fork, wait
import copy
# import seaborn as sns


class same_ex_model(nn.Module):
    def name(self):
        return 'ColorHistogram_Model'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = opt.checkpoints_dir
        self.MSE=nn.MSELoss()
        self.L1 = nn.L1Loss()
        nb = opt.batchSize
        # size = opt.fineSize 

        self.img_type = opt.img_type
        self.pad = 30
        self.reppad = nn.ReplicationPad2d(self.pad)

        # self.extructor_d=SimpleResNet(3,3).cuda()
        # self.extructor_c=SimpleResNet(3,3).cuda()
        self.extructor_fuse=SaMam(
            nVSSMs=opt.nVSSMs,
            nSAVSSMs=opt.nSAVSSMs,
            nSAVSSGs=opt.nSAVSSGs,

            embed_dim=opt.embed_dim,
            patch_size=opt.patch_size,
            representation_dim=opt.representation_dim,
            d_state=opt.d_state,
            expand=opt.expand,
            compress_ratio=opt.compress_ratio,
            squeeze_factor=opt.squeeze_factor,
            mamba_from_trion=opt.mamba_from_trion
        ).cuda()
        # self.extructor_fuse=color_fuse_net_EMMA(6,3).cuda()
        # self.extructor_fuse=color_fuse_net_DNCM(6,3).cuda()


        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            ch=torch.load(opt.ckpt_name)
            # self.extructor_d.load_state_dict(ch['net_extructor_d'])
            # self.extructor_c.load_state_dict(ch['net_extructor_c'])
            self.epoch=ch['epoch']
            self.optimizer=ch['optimizer']
            self.scheduler = ch['scheduler']
            self.extructor_fuse.load_state_dict(ch['net_extructor_fuse'])

    def net_train(self):
        # self.extructor_d.train()
        # self.extructor_c.train()
        self.extructor_fuse.train()
        if self.opt.continue_train==True:
            return self.optimizer,self.scheduler,self.epoch


    def net_eval(self):
        # self.extructor_d.eval()
        # self.extructor_c.eval()
        self.extructor_fuse.eval()

    def set_input(self, input):    
        self.img1_A=input['img1_A'].cuda()
        self.img1_B=input['img1_B'].cuda()
        self.img2_A=input['img2_A'].cuda()
        self.img2_B=input['img2_B'].cuda()
    
    def forward(self):
        #融合
        # Ics = []
        # for i in range(0, self.img2_A.shape[0]):
        #     content_i = self.img2_A[i:i + 1, :, :, :]
        #     style_i = self.img1_B[i:i + 1, :, :, :]
        #     output_i = self.extructor_fuse(content_i, style_i)
        #     Ics.append(output_i)
        #     torch.cuda.empty_cache()
        # self.ex_rec_1 = torch.cat(Ics, 0)
        #
        # Ics = []
        # for i in range(0, self.img2_A.shape[0]):
        #     content_i = self.img1_A[i:i + 1, :, :, :]
        #     style_i = self.img2_B[i:i + 1, :, :, :]
        #     output_i = self.extructor_fuse(content_i, style_i)
        #     Ics.append(output_i)
        #     # torch.cuda.empty_cache()
        # self.ex_rec_2 = torch.cat(Ics, 0)

        Ics = [self.extructor_fuse(self.img2_A[i:i + 1], self.img1_B[i:i + 1]) for i in range(self.img2_A.shape[0])]
        # torch.cuda.empty_cache()
        self.ex_rec_1 = torch.stack(Ics, dim=0).squeeze(1)  # shape: [B, C, H, W]

        # Ics = [self.extructor_fuse(self.img1_A[i:i + 1], self.img2_B[i:i + 1]) for i in range(self.img2_A.shape[0])]
        # self.ex_rec_2 = torch.stack(Ics, dim=0).squeeze(1)  # shape: [B, C, H, W]

        # Ics = [self.extructor_fuse(self.img1_B[i:i + 1], self.img2_A[i:i + 1]) for i in range(self.img2_A.shape[0])]
        # self.ex_rec_2 = torch.stack(Ics, dim=0).squeeze(1)  # shape: [B, C, H, W]

        # Ics1 = []
        # for i in range(self.img2_A.shape[0]):
        #     content_i = self.img2_A[i:i + 1]
        #     style_i = self.img1_B[i:i + 1]
        #     Ics1.append(fork(self.extructor_fuse, content_i, style_i))
        #
        #
        # Ics2 = []
        # for i in range(self.img2_A.shape[0]):
        #     content_i = self.img1_A[i:i + 1]
        #     style_i = self.img2_B[i:i + 1]
        #     Ics2.append(fork(self.extructor_fuse, content_i, style_i))
        #
        # Ics1 = [wait(f) for f in Ics1]
        # self.ex_rec_1 = torch.cat(Ics1, dim=0)
        # Ics2 = [wait(f) for f in Ics2]
        # self.ex_rec_2 = torch.cat(Ics2, dim=0)

        # self.ex_rec_1 = self.extructor_fuse(self.img2_A, self.img1_B)
        # self.ex_rec_2 = self.extructor_fuse(self.img1_A, self.img2_B)

    def get_current_losses(self):
        # loss1=self.L1(self.img1_A_d, self.img2_A_d)+self.L1(self.img1_B_d, self.img2_B_d)
        # loss2=self.L1(self.ex_rec_1, self.img2_A)
        # loss3=self.L1(self.ex_rec_2, self.img1_A)

        loss2=self.L1(self.ex_rec_1, self.img1_A)
        # loss3=self.L1(self.ex_rec_2, self.img2_A)

        # loss4=self.L1(self.ex_rec_1, self.img2_A)
        # loss5=self.L1(self.ex_rec_2, self.img1_A)
        # loss4=self.MSE(self.ex_rec_1, self.img1_A)
        # loss5=self.MSE(self.ex_rec_2, self.img2_A)
        # return (loss2+loss3)+(loss4+loss5)/2
        return loss2

    def save_networks(self,epoch,optimizer,scheduler):
        checkpoint = {
            'epoch': epoch,
            # 'net_extructor_d': self.extructor_d.state_dict(),
            # 'net_extructor_c': self.extructor_c.state_dict(),
            'net_extructor_fuse': self.extructor_fuse.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(checkpoint, f'{self.opt.checkpoints_dir}/epoch_{epoch}.pth')
        # torch.save(checkpoint, f'{self.opt.checkpoints_dir}/epoch_{epoch}.pth')

    def pad_t(self, tensor, divisor=128):
        _, _, h, w = tensor.shape
        pad_w = (divisor - w % divisor) % divisor  # 右侧填充宽度
        pad_h = (divisor - h % divisor) % divisor  # 底部填充高度
        # pad参数格式：(左, 右, 上, 下)
        # padded_tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_tensor = F.interpolate(tensor, size=(h+pad_h,w+pad_w),mode='bicubic',align_corners=False)

        return padded_tensor

    def test(self, inp, tar):
        self.net_eval()
        _,_,w,h=inp.shape
        # print(w,h)
        inp=self.pad_t(inp)
        # tar=self.pad_t(tar)

        # tar=self.pad_t(tar)
        # tar = F.interpolate(inp, size=inp.shape[2:],mode='bicubic',align_corners=False)
        # tar = F.interpolate(inp, size=(64,64),mode='bicubic',align_corners=False)

        # tar=self.pad_t(tar)
        with torch.no_grad():
            # im1_d=self.extructor_d(inp)
            # im1_c=self.extructor_c(tar)
            #融合
            # inp_fuse = torch.cat((im1_d,im1_c),1)
            out=self.extructor_fuse(inp,tar)
        out= F.interpolate(out, size=(w,h),mode='bicubic',align_corners=False)

        return out[:,:,:w,:h]


    def get_current_visuals(self):
        d={}
        # d['ex_d_img1_B_d']=self.img1_B_d
        d['img1_A']=self.img1_A/2+0.5
        d['img1_B']=self.img1_B/2+0.5
        d['img2_A']=self.img2_A/2+0.5
        d['out']=self.ex_rec_1/2+0.5
        
        # d['ex_d_img1_B_d']=self.img1_B_d
        # d['ex_d_img2_B_d']=self.img2_B_d

        return d

