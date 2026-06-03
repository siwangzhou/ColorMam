import os
from PIL import Image

# # 原始图片文件夹路径
input_folder = 'D:\code\Python\project\mechine Learning\Data/val_data/test_final\o'  # 替换为你的路径
# 输出文件夹路径
output_folder = 'D:\code\Python\project\mechine Learning\Data/val_data/test_final/input'  # 替换为你的路径

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 获取图片文件列表（只获取常见图片类型）
image_files = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
])

counter = 0
repeat_times = 21

for i in range(repeat_times):
    j=0
    for file in image_files:
        if i==j:
            j += 1
            continue
        img_path = os.path.join(input_folder, file)
        img = Image.open(img_path)
        w,h=img.size
        m=min(w,h)
        img=img.crop((0,0,m,m))
        img=img.resize((1024,1024),Image.BICUBIC)
        # 保存为 png，使用递增数字命名
        output_path = os.path.join(output_folder, f"{counter}.png")
        img.save(output_path, format='PNG')
        j += 1
        counter += 1





input_folder = 'D:\code\Python\project\mechine Learning\Data/val_data/test_final\o'  # 替换为你的路径
# 输出文件夹路径
output_folder = 'D:\code\Python\project\mechine Learning\Data/val_data/test_final/target'  # 替换为你的路径
# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 读取并排序所有图片文件（支持常见格式）
image_files = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
])

repeat_times = 20
file_index = 0  # 全局命名编号
# j=0
for img_file in image_files:
    img_path = os.path.join(input_folder, img_file)
    img = Image.open(img_path)
    w, h = img.size
    m=min(w,h)
    img=img.crop((0,0,m,m))
    img = img.resize((1024, 1024), Image.BICUBIC)

    for i in range(repeat_times):
        # if i==j:
        #     continue
        output_path = os.path.join(output_folder, f"{file_index}.png")
        img.save(output_path, format='PNG')
        file_index += 1


# folder = 'D:\code\Python\project\mechine Learning\Data/val_data\jianzhu\input'  # 替换为你的路径
#
# # 遍历所有图片文件
# for filename in os.listdir(folder):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#         img_path = os.path.join(folder, filename)
#
#         # 打开图像
#         with Image.open(img_path) as img:
#             # 计算新尺寸（缩小2倍）
#             new_size = (img.width // 2, img.height // 2)
#             resized_img = img.resize(new_size, resample=Image.LANCZOS)
#
#             # 保存为 PNG，使用原始文件名但统一扩展名为 .png
#             name_wo_ext = os.path.splitext(filename)[0]
#             output_path = os.path.join(folder, f"{name_wo_ext}.png")
#             resized_img.save(output_path, format='PNG')


# import os
# from PIL import Image
# from concurrent.futures import ThreadPoolExecutor
# from functools import partial
#
# # 输入和输出文件夹
# input_folder = 'D:\code\Python\project\mechine Learning\Data\LIU2K'
# output_folder = 'D:\code\Python\project\mechine Learning\Data\LR_color_transfer'
# os.makedirs(output_folder, exist_ok=True)
#
# # 获取所有图像文件路径
# image_files = [
#     f for f in os.listdir(input_folder)
#     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
# ]
#
# def process_image(filename, input_folder, output_folder):
#     input_path = os.path.join(input_folder, filename)
#     name_wo_ext = os.path.splitext(filename)[0]
#     output_path = os.path.join(output_folder, f"{name_wo_ext}.png")
#
#     try:
#         with Image.open(input_path) as img:
#             # n=img.height/
#             new_size = (img.width // 3, img.height // 3)
#             resized = img.resize(new_size, resample=Image.BICUBIC)
#             resized.save(output_path, format='PNG')
#         print(f"✔ Saved: {output_path}")
#     except Exception as e:
#         print(f"❌ Error processing {filename}: {e}")
#
# # 用线程池并发处理图像
# with ThreadPoolExecutor(max_workers=8) as executor:
#     executor.map(partial(process_image, input_folder=input_folder, output_folder=output_folder), image_files)

