import os
import cv2
from utils import make_directory

# Code for generating dehaze images by CLAHE (in order to calculate Loss_CLAHE)
if __name__ == '__main__':

    folder_list = ['Hidden', 'RESIDE_RTTS']

    for folder_name in folder_list:

        data_path = '/opt/ml/final-project-level3-cv-17/data/' + folder_name +'/hazy'
        out_path = '/opt/ml/final-project-level3-cv-17/data/' + folder_name + '/gt_clahe'
        make_directory(out_path)
        name_list = list(os.walk(data_path))[0][2]
        for i, name in enumerate(name_list):
            img = cv2.imread(os.path.join(data_path,name))
            b,g,r = cv2.split(img)
            img_rgb = cv2.merge([r,g,b])
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(16,16))
            img_rgb2 = clahe.apply(img_rgb.reshape(-1)).reshape(img_rgb.shape)
            r, g, b = cv2.split(img_rgb2)
            img_out = cv2.merge([b, g, r])
            cv2.imwrite(os.path.join(out_path,name), img_out)
