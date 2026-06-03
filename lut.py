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
# import colour
from copy import deepcopy
from torchvision import transforms


def load_cube_lut(lut_file):
    with open(lut_file, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('#')]  # 跳过注释行

    # 解析LUT尺寸（更健壮的解析方式）
    size = None
    for line in lines:
        if line.lower().startswith('title') or line.lower().startswith('lut_3d_size'):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    size = int(parts[-1])
                    break
                except ValueError:
                    continue
    if size is None:
        raise ValueError("无法解析LUT尺寸")

    # 读取LUT数据（处理不同格式）
    lut_data = []
    for line in lines:
        if len(line.split()) == 3:
            try:
                r, g, b = map(float, line.split())
                lut_data.append([r, g, b])
            except ValueError:
                continue

    if len(lut_data) != size ** 3:
        raise ValueError(f"LUT数据量不匹配，预期{size ** 3}个值，实际得到{len(lut_data)}")

    lut_data = np.array(lut_data, dtype=np.float32)
    return lut_data.reshape((size, size, size, 3))


import numpy as np


def apply_cube_lut(image, lut_data):
    size = lut_data.shape[0]
    coords = image * (size - 1)

    # 计算所有像素的低位和高位坐标
    low = np.floor(coords).astype(int)
    high = np.ceil(coords).astype(int)
    delta = coords - low

    # 边界保护（向量化实现）
    low = np.clip(low, 0, size - 2)
    high = np.clip(high, 1, size - 1)

    # 获取8个顶点数据（一次性提取所有像素所需数据）
    c000 = lut_data[low[..., 0], low[..., 1], low[..., 2]]
    c001 = lut_data[low[..., 0], low[..., 1], high[..., 2]]
    c010 = lut_data[low[..., 0], high[..., 1], low[..., 2]]
    c011 = lut_data[low[..., 0], high[..., 1], high[..., 2]]
    c100 = lut_data[high[..., 0], low[..., 1], low[..., 2]]
    c101 = lut_data[high[..., 0], low[..., 1], high[..., 2]]
    c110 = lut_data[high[..., 0], high[..., 1], low[..., 2]]
    c111 = lut_data[high[..., 0], high[..., 1], high[..., 2]]

    # 三线性插值计算（完全向量化）
    delta_z = delta[..., 2, None]  # 增加维度用于广播
    c00 = c000 * (1 - delta_z) + c001 * delta_z
    c01 = c010 * (1 - delta_z) + c011 * delta_z
    c10 = c100 * (1 - delta_z) + c101 * delta_z
    c11 = c110 * (1 - delta_z) + c111 * delta_z

    delta_y = delta[..., 1, None]
    c0 = c00 * (1 - delta_y) + c01 * delta_y
    c1 = c10 * (1 - delta_y) + c11 * delta_y

    delta_x = delta[..., 0, None]
    output = c0 * (1 - delta_x) + c1 * delta_x

    return np.clip(output, 0, 1)


from multiprocessing import Pool, cpu_count


def process_single_task(args):
    """多进程任务处理单元"""
    img_path, lut, output_base, lut_path = args
    try:
        # 创建图片专属目录
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        output_dir = os.path.join(output_base, img_name)
        os.makedirs(output_dir, exist_ok=True)

        # 加载图片
        image = cv2.imread(img_path)/255
        
        if image is None:
        	raise ValueError(f"图片加载失败，请检查路径是否正确: {img_path}")
        #print(image.shape)
        
        # 获取原始宽高
        # height, width = image.shape[:2]

	# 找出最小边和最大边
    #     min_side = min(width, height)
    #     max_side = max(width, height)
	
        # 目标最小边大小
        # target_min_side = 768

	# 计算缩放比例
    #     scale = target_min_side / min_side
	
	# 计算新的宽高
    #     new_width = int(width * scale)
    #     new_height = int(height * scale)

	# 使用双三次插值进行resize
    #     image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        if image is None:
            raise ValueError(f"图片加载失败: {img_path}")
        # image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2RGB)

        # 加载LUT
        # lut_data = load_cube_lut(lut_path)
        # lut_data = lut
        # print('------------------')

        # 应用LUT
        processed = apply_cube_lut(image, lut)
        processed = (processed * 255).astype(np.uint8)
        processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

        # 生成输出路径
        lut_name = os.path.splitext(os.path.basename(lut_path))[0]
        output_path = os.path.join(output_dir, f"{lut_name}.jpg")

        cv2.imwrite(output_path, processed, [cv2.IMWRITE_JPEG_QUALITY,97])
        return True
    except Exception as e:
        print(f"faile: {img_path} + {lut_path}\n错误信息: {str(e)}")
        return False


def batch_process(config):
    """主处理函数"""
    # 获取所有图片和LUT路径
    img_dir = config['image_dir']
    lut_dir = config['lut_dir']
    output_dir = config['output_dir']

    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))][3550:]
    lut_paths = [os.path.join(lut_dir, f) for f in os.listdir(lut_dir)
                 if f.lower().endswith(('.cube', '.3dl'))]
    lut_loads=[]
    for i in lut_paths:
        lut_loads.append(load_cube_lut(i))

    # 生成任务参数列表
    tasks = [(img, lut, output_dir,lut_path) for img in img_paths for lut,lut_path in zip(lut_loads, lut_paths) ]

    # 配置进程池
    num_processes = min(cpu_count(), len(tasks))  # 根据网页1建议动态调整进程数
    with Pool(processes=12) as pool:
        results = pool.imap_unordered(process_single_task, tasks)

        # 进度跟踪
        success = 0
        for i, result in enumerate(results, 1):
            if result: success += 1
            if i % 10 == 0:
                print(f"处理进度: {i}/{len(tasks)}，成功率: {success / i:.1%}")


if __name__ == "__main__":
    # config = {
    #     'image_dir': r"/media/good-person/新加卷/code/Python/project/mechine_Learning/Data/LR_color_transfer",  # 图片目录
    #     'lut_dir': r"/media/good-person/新加卷/code/Python/project/mechine_Learning/Data/3dLut",  # LUT目录
    #     'output_dir': r"/media/good-person/新加卷/code/Python/project/mechine_Learning/Data/colorTransfer"  # 输出根目录
    # }

    config = {
        'image_dir': r"D:\code\Python\project\mechine_Learning\Data/LR_color_transfer",  # 图片目录
        'lut_dir': r"D:\code\Python\project\mechine_Learning\Data/3dLut",  # LUT目录
        'output_dir': r"D:\code\Python\project\mechine_Learning\Data/colorTransfer"  # 输出根目录
    }

    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)

    # 启动处理流程
    batch_process(config)


# import cv2
# import numpy as np
# from colour.io import read_LUT_IridasCube
# from colour import LUT3D
# from colour.algebra import LinearInterpolator  # 导入插值器
#
# # 1. 读取LUT文件
# cube_path = r"E:\python\project\dataset/3d_lut\Titanium_Cinematic_02.cube"
# lut_data = read_LUT_IridasCube(cube_path).table
#
# # 2. 创建LUT3D对象（新版API无需在初始化时指定插值器）
# lut_3d = LUT3D(
#     table=lut_data,
#     name="Custom LUT",
#     domain=np.array([[0, 0, 0], [1, 1, 1]])
# )
#
# # 3. 读取图像
# image_bgr = cv2.imread(r"E:\shipai8bit-2/ref/1x_ds/00000_IMG_20250403_161457.jpg")
# image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) / 255.0
#
# # 4. 应用LUT（关键修正：使用正确插值参数）
# output_rgb = lut_3d.apply(
#     image_rgb,
#     interpolator=LinearInterpolator,  # 传入插值器类
#     interpolator_kwargs={"method": "trilinear"}  # 指定三线性插值
# )
#
# # 5. 保存结果
# output_bgr = cv2.cvtColor((output_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
# cv2.imwrite(r'E:\shipai8bit-2\output\a.png', output_bgr)
