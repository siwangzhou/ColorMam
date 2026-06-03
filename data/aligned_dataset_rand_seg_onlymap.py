import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, get_transform_lab, no_transform
from data.image_folder import make_dataset
from data.datatransfer import MyCustomTransform
from PIL import Image,ImageOps
import random
import torch
# import random
import numpy as np
from torch import nn

class SyncRandomFlip:
    def __init__(self):
        self.h_flip = random.random() < 0.5  # 随机水平翻转决策
        self.v_flip = random.random() < 0.5  # 随机垂直翻转决策

    def __call__(self, img):
        if self.h_flip:
            img = transforms.functional.hflip(img)
        if self.v_flip:
            img = transforms.functional.vflip(img)
        return img

class AlignedDataset_Rand_Seg_onlymap(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.img_type = opt.img_type

        self.dir_A = opt.dataroot
        # self.A_paths = make_dataset(self.dir_A)
        self.A_paths = [os.path.join(self.dir_A, d) for d in os.listdir(self.dir_A) if os.path.isdir(os.path.join(self.dir_A, d))][:4400]
        # random.shuffle(self.A_paths)
        print('--------------------',len(self.A_paths))
        self.A_paths = sorted(self.A_paths)
        self.A_path_arr=[]
        for i in self.A_paths:
            # print(i)
            # if len(make_dataset(i))>1:
            self.A_path_arr.append(make_dataset(i))
        # print(self.A_path_arr)
        # print(self.A_paths)
        self.A_size = len(self.A_paths)
        self.rand_crop=transforms.RandomCrop(opt.fineSize)
        self.train_transform = transforms.Compose([
            MyCustomTransform(hue_ranges=None, saturation_ranges=None, light_ranges=None,
                              temp_range=(-500, 500)),  # 基于HSV的色调色温范围
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),  # 整体颜色抖动
            # transforms.RandomGrayscale(p=0.01),  # 转灰度图
            transforms.ToTensor()
        ])
        

    def __getitem__(self, index):
        if len(self.A_path_arr[index % self.A_size])==0:
            print(self.A_paths[index % self.A_size])
        A_paths = self.A_path_arr[index % self.A_size]
        random.shuffle(A_paths)
        # print(A_paths[0])
        img1=Image.open(A_paths[0]).convert('RGB')
        img2=Image.open(A_paths[1]).convert('RGB')
        W,H=img1.size
        m=max(W,H)
        if m==W:
            k=W/H
            H=1024
            W=H*k
        else:
            k = H / W
            W = 1024
            H = W * k

        img1=img1.resize((int(W/1),int(H/1)),Image.BICUBIC)
        img2=img2.resize((int(W/1),int(H/1)),Image.BICUBIC)

        # img2=img2.resize()
        # img1=self.rand_crop(img)
        # img2=self.rand_crop(img)

        img1 = self.train_transform(img1)
        img2 = self.train_transform(img2)
        _, H, W = img1.shape
        h=self.opt.fineSize
        w=self.opt.fineSize

        top = torch.randint(0, H - h + 1, (1,)).item()
        left = torch.randint(0, W - w + 1, (1,)).item()
        img1_A = img1[:, top:top+h, left:left+w]
        img2_A = img2[:, top:top+h, left:left+w]
        
        p=0.5
        # 水平翻转
        if random.random() < p:
            img1_A = torch.flip(img1_A, dims=[2])
            img2_A = torch.flip(img2_A, dims=[2])
        # 垂直翻转
        if random.random() < p:
            img1_A = torch.flip(img1_A, dims=[1])
            img2_A = torch.flip(img2_A, dims=[1])

        top = torch.randint(0, H - h + 1, (1,)).item()
        left = torch.randint(0, W - w + 1, (1,)).item()
        img1_B = img1[:, top:top+h, left:left+w]
        img2_B = img2[:, top:top+h, left:left+w]

        # 水平翻转
        if random.random() < p:
            img1_B = torch.flip(img1_B, dims=[2])
            img2_B = torch.flip(img2_B, dims=[2])
        # 垂直翻转
        if random.random() < p:
            img1_B = torch.flip(img1_B, dims=[1])
            img2_B = torch.flip(img2_B, dims=[1])

        return {'img1_A': img1_A*2-1, 'img1_B': img1_B*2-1, 'img2_A': img2_A*2-1, 'img2_B': img2_B*2-1}

    def __len__(self):
        # return 10000
        return self.A_size

    def name(self):
        return 'AlignedDataset_Rand_Seg_onlymap'

class val_data_loader(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.valroot = opt.val_dataroot

        dir_input = self.valroot+'/'+'input'
        dir_target = self.valroot+'/'+'target'
        self.input_paths = make_dataset(dir_input)
        self.target_paths = make_dataset(dir_target)
        self.input_paths = sorted(self.input_paths)
        self.target_paths = sorted(self.target_paths)[:]
        # print(self.input_paths)
        # print(self.target_paths)
        self.A_size = len(self.target_paths)

        self.train_transform = transforms.Compose([
            MyCustomTransform(hue_ranges=None, saturation_ranges=None, light_ranges=None,
                              temp_range=(-500, 500)),  # 基于HSV的色调色温范围
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),  # 整体颜色抖动
            # transforms.RandomGrayscale(p=0.01),  # 转灰度图
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        input_path = self.input_paths[index % self.A_size]
        target_path = self.target_paths[index % self.A_size]
        # print(input_path)
        # print(target_path)
        inp=ImageOps.exif_transpose(Image.open(input_path)).convert('RGB')
        tar=Image.open(target_path).convert('RGB')

        # tar = self.train_transform(inp)

        inp=transforms.ToTensor()(inp)
        tar=transforms.ToTensor()(tar)

        # print(self.input_paths[index % self.A_size].split('/')[-1].split('.')[0])

        return {'input': inp, 'target': tar, 'name': self.input_paths[index % self.A_size].split('/')[-1].split('.')[0]}
        # return {'input': tar, 'target': inp, 'name': self.input_paths[index % self.A_size].split('/')[-1].split('.')[0]}

    def __len__(self):
        # return 1
        return self.A_size

    def name(self):
        return 'val_data_loader'
