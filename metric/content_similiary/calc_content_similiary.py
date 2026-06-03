import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity

import torch
from torchvision import transforms

import ldc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ldc_model = ldc.LDC()
ldc_model.load_state_dict(torch.load('./ldc.pth', map_location='cpu'))
ldc_model.to(device).eval()


def calculate_ldc_edge(image_path):
    image = Image.open(image_path).convert('RGB')

    h, w = image.size
    h = int(h - h % 32)
    w = int(w - w % 32)

    mean = torch.tensor([103.939, 116.779, 123.68]).to(device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    image = transforms.functional.resize(image, (w, h))
    image = transforms.functional.to_tensor(image)[None, ...].to(device) * 255

    edges = ldc_model(image - mean)
    avg_edge = ldc.postprocess_edges(edges)
    avg_edge = torch.from_numpy(avg_edge).unsqueeze(0).unsqueeze(0) / 255

    return avg_edge


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-folder', type=str, default='../../results/DLut')
    # parser.add_argument('--result-folder', type=str, default='D:\code\Python\project\mechine Learning\colorTransfer\D-LUT-main\D-LUT-main\Imgs/result')
    # parser.add_argument('--result-folder', type=str, default='../../flows')
    # parser.add_argument('--content-folder', type=str, default='D:\code\Python\project\mechine Learning\Data/val_data/test_data/input')
    parser.add_argument('--content-folder', type=str, default='D:/code/Python/project/mechine_Learning/Data/val_data/test_mix/input')

    args = parser.parse_args()

    # check cmd arguments
    if not os.path.exists(args.result_folder):
        print('Cannot find the result folder: {0}'.format(args.result_folder))
        exit()
    if not os.path.exists(args.content_folder):
        print('Cannot find the content folder: {0}'.format(args.content_folder))
        exit()

    result_files = os.listdir(args.result_folder)
    content_files = os.listdir(args.content_folder)
    assert len(result_files) == len(content_files)

    print('--------------------------------------------------------------------------------')
    print('result folder: {0}'.format(args.result_folder))
    print('style folder: {0}'.format(args.content_folder))
    print('--------------------------------------------------------------------------------')

    # calculate content ssim score
    content_similiary_scores = []
    pbar = tqdm(result_files, total=len(result_files), unit='file')
    for idx, fname in enumerate(pbar):

        # load input image and content image
        result_path = os.path.join(args.result_folder, fname)
        content_path = os.path.join(args.content_folder, fname)
        
        result_edge = calculate_ldc_edge(result_path)
        content_edge = calculate_ldc_edge(content_path)

        # calculate edges
        result_edge = result_edge[0].permute(1, 2, 0).cpu().numpy()
        content_edge = content_edge[0].permute(1, 2, 0).cpu().numpy()
        # print(content_edge)
        score = structural_similarity(result_edge, content_edge, multichannel=True,channel_axis=-1,data_range=1)

        # save score and update pbar
        content_similiary_scores.append(score)
        pbar.set_description('Avg Content Similiary Score: {0:.4f}'.format(np.mean(np.asarray(content_similiary_scores))))

print('--------------------------------------------------------------------------------')
print('Final Avg Content Similiary Score: {0:.4f}'.format(np.mean(np.asarray(content_similiary_scores))))
