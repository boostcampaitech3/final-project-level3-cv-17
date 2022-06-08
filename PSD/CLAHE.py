import os
import cv2
from utils import make_directory

# Code for generating dehaze images by CLAHE (in order to calculate Loss_CLAHE)
if __name__ == '__main__':

    clip_limit = 1
    grid_size = 64
    folder_name = f'/gt_clahe_{clip_limit}_{grid_size}'

    # data_path = '/opt/ml/input/final-project-level3-cv-17/data/RESIDE_RTTS/hazy'
    # out_path = f'/opt/ml/input/final-project-level3-cv-17/data/RESIDE_RTTS/gt_clahe_{clip_limit}_{grid_size}'

    data_path = '/opt/ml/input/final-project-level3-cv-17/PSD/output/pretrained_model/finetune/Hidden/'
    weight_name = data_path.split('/')[-3]
    data_name = data_path.split('/')[-2]
    out_path = f'clahe/{weight_name}/{data_name}/{clip_limit}_{grid_size}'
    make_directory(out_path)

    name_list = sorted(list(os.walk(data_path))[0][2])
    for i, name in enumerate(name_list):
        img = cv2.imread(os.path.join(data_path,name))
        b,g,r = cv2.split(img)
        img_rgb = cv2.merge([r,g,b])
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size,grid_size))
        img_rgb2 = clahe.apply(img_rgb.reshape(-1)).reshape(img_rgb.shape)
        r, g, b = cv2.split(img_rgb2)
        img_out = cv2.merge([b, g, r])
        cv2.imwrite(os.path.join(out_path,name), img_out)
