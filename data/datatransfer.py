import glob
import os
# from pylut import *
# import pylut
import cv2
# import kornia
import torch
import PIL.Image as Image
import numpy as np
import random
from copy import deepcopy
from torchvision import transforms
class MyCustomTransform:
    def __init__(self, hue_ranges=None, saturation_ranges=None, light_ranges=None, temp_range=None):
        """初始化参数"""
        self.hue_ranges=hue_ranges
        self.saturation_ranges = saturation_ranges
        self.light_ranges = light_ranges
        self.temp_range = temp_range

    def __call__(self, img):
        """
        输入: PIL Image
        输出: 处理后的 PIL Image 或 Tensor
        """
        # 你的处理逻辑
        processed_img = self._apply_effect(img)
        return processed_img

    def _apply_effect(self, img):

        # lut = Lut.from_file("E:\python\project\dataset/3d_lut\Titanium_Cinematic_01.cube")
        # """实际处理函数（示例：随机色相偏移）"""
        # 转换为HSV颜色空间
        opencv_img = np.array(img)
        # 处理不同通道情况
        hsv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2HSV)
        # print(hsv_img.shape)
        h, s, v = cv2.split(hsv_img)
        # print(s.shape)
        # hsv_img = img.convert('HSV')
        # h, s, v = hsv_img.split()
        # h.show()
        # s.show()
        # v.show()
        h_array = np.array(h, dtype=np.float32) / 180.0  # 标准化到0-1
        # print(h_array.shape)
        # h_array = cv2.GaussianBlur(h_array, (7, 7), 2)
        h_array_temp = deepcopy(h_array)
        s_array = np.array(s, dtype=np.float32) / 255.0
        s_array_temp = deepcopy(s_array)
        v_array = np.array(v, dtype=np.float32) / 255.0
        v_array_temp = deepcopy(v_array)

        # 定义HSV颜色范围（所有范围标准化到0-1）
        color_definitions = {
            # 'red': ((0, 20), (0.5, 1)),  # 0-20°
            'red': ((310 - 359, 20), (0.5, 1)),  # 0-20°
            # 'orange': ((21, 50), (1.0, 0.16)),  # 20-40°
            # 'yellow': ((51, 68), (1.0, 0.24)),  # 40-60°
            'orange': ((21, 68), (1.0, 0.16)),  # 20-40°
            'green': ((69, 154), (1.0, 0.55)),  # 60-140°
            'blue': ((155, 199), (1.0, 0.78)),  # 140-200°
            'purple': ((200, 250), (1.0, 1.0)),  # 200-260°
            'pink': ((251, 310), (1.0, 1.0)),  # 260-300°
            # 'red2': ((311, 360), (1.0, 1.0))  # 300-360°
        }

        # 随机选择0-4个颜色进行处理
        selected_colors = random.sample(list(color_definitions.items()), k=random.randint(3, 6))

        for color_name, (hue_range, (v_min, v_max)) in selected_colors:
        # 对每种颜色进行处理

            lower_h, upper_h = hue_range
            # lower_h-=10
            # upper_h+=10
            # 创建颜色掩码
            if lower_h >= 0:
                mask = (h_array >= lower_h/360) & (h_array < upper_h/360)
            else:
                # 处理红色范围跨越0°的情况
                # print(123)
                mask = (h_array >= - lower_h/360)  |(h_array < upper_h/360)
                # print(mask .tolist())
            # 添加饱和度(S)和明度(V)阈值条件
            # print(s_array)
            mask = mask & (s_array >= 43 / 255) & (v_array >= 46 / 255)
            # mask = mask & (s_array >= 43 / 255)& (s_array <221 / 255) & (v_array >= 46 / 255)& (v_array <221 / 255)
            # 应用色相偏移（如果有设置）
            if self.hue_ranges != None:
                min_h_shift, max_h_shift = self.hue_ranges
                h_shift = random.uniform(min_h_shift, max_h_shift)  # 已经是0-1范围
                h_array_temp[mask] = (h_array[mask] + h_shift)*180 % 180/180
                # print(mask.tolist())
            # 应用饱和度偏移（如果有设置）
            if self.saturation_ranges != None:
                min_s_shift, max_s_shift = self.saturation_ranges
                s_shift = random.uniform(min_s_shift, max_s_shift)
                # s_array = np.clip(s_array * s_shift, 0, 1)
                s_array_temp[mask] = np.clip(s_array * s_shift, 0, 1)[mask]

            if self.light_ranges != None:
                min_v_shift, max_v_shift = self.light_ranges
                v_shift = random.uniform(min_v_shift, max_v_shift)
                # v_array_temp[mask] = (v_array[mask] + v_shift)
                v_array_temp[mask] = np.clip(v_shift * v_array, 0, 1)[mask]


        mask_l = (s_array_temp < 0)
        mask_r = (s_array_temp > 1)
        s_array_temp[mask_l] = 0
        s_array_temp[mask_r] = 1

        mask_l = (v_array_temp < 0)
        mask_r = (v_array_temp > 1)
        v_array_temp[mask_l] = 0
        v_array_temp[mask_r] = 1

        # #高斯模糊，使边缘更平滑
        # h_array_temp = cv2.GaussianBlur(h_array_temp, (7, 7), 2)
        # 还原到0-179范围（OpenCV格式）
        # h_array_temp = np.mod(h_blur, 180).astype(np.uint8)

        # 将数据转换回0-255范围
        h_array = (h_array_temp * 255).astype(np.uint8)
        s_array = (s_array_temp * 255).astype(np.uint8)
        v_array = (v_array_temp * 255).astype(np.uint8)

        # hsv_img = cv2.merge([h_array, s_array, v_array])  # 通道顺序H,L,S
        # rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HLS2RGB)/255
        # result=Image.fromarray(rgb_img)
        # 转换回PIL图像
        h_shifted = Image.fromarray(h_array, 'L')
        # h=Image.fromarray(h,'L')
        # h.show()
        s_shifted = Image.fromarray(s_array, 'L')
        v_img = Image.fromarray(v_array, 'L')
        shifted_hsv = Image.merge('HSV', (h_shifted, s_shifted, v_img))
        # 转换回RGB并保存
        result = shifted_hsv.convert('RGB')

        if self.temp_range != None:
            # 随机生成色温值
            temperature = random.uniform(*self.temp_range)
            # 分离RGB通道
            r, g, b = result.split()
            # 应用色温调整
            if temperature > 0:  # 暖色调
                r = r.point(lambda x: min(255, x * (1 + temperature / 4000)))
                g = g.point(lambda x: min(255, x * (1 + temperature / 6000)))
                b = b.point(lambda x: max(0, x * (1 - temperature / 3000)))
            else:  # 冷色调
                temperature = abs(temperature)
                r = r.point(lambda x: max(0, x * (1 - temperature / 4000)))
                g = g.point(lambda x: max(0, x * (1 - temperature / 6000)))
                b = b.point(lambda x: min(255, x * (1 + temperature / 3000)))
            # 合并通道并保存
            result = Image.merge('RGB', (r, g, b))
        return result