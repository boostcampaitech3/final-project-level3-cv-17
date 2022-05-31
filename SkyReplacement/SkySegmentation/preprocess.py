import cv2
import glob
import os
from tqdm import tqdm


os.remove('/opt/ml/input/final-project-level3-cv-17/data/skyseg/img_dir/train/ADE_train_00007341.jpg')
os.remove('/opt/ml/input/final-project-level3-cv-17/data/skyseg/img_dir/val/ADE_val_00000424.jpg')

mask_paths_list_train = glob.glob('/opt/ml/input/final-project-level3-cv-17/data/skyseg/ann_dir/train/*')
mask_paths_list_val = glob.glob('/opt/ml/input/final-project-level3-cv-17/data/skyseg/ann_dir/val/*')
mask_paths_list = mask_paths_list_train+mask_paths_list_val

img_paths_list_train = glob.glob('/opt/ml/input/final-project-level3-cv-17/data/skyseg/img_dir/train/*')
img_paths_list_val = glob.glob('/opt/ml/input/final-project-level3-cv-17/data/skyseg/img_dir/val/*')
img_path_list =img_paths_list_train+img_paths_list_val

# ann_dir -> img_dir
# png to jpg

for mask_path in tqdm(mask_paths_list):
    img = cv2.imread(mask_path)
    if 255 in img :
        pass
    else:
        os.remove(mask_path)
        img_path = mask_path.replace('ann_dir','img_dir').replace('png','jpg').replace('dg_train','train').replace('dg_val','val')
        os.remove(img_path)

import numpy as np

mask_paths_list_train = glob.glob('/opt/ml/input/final-project-level3-cv-17/data/skyseg/ann_dir/train/*')
mask_paths_list_val = glob.glob('/opt/ml/input/final-project-level3-cv-17/data/skyseg/ann_dir/val/*')
mask_paths_list = mask_paths_list_train+mask_paths_list_val

for mask_path in tqdm(mask_paths_list):
    img = cv2.imread(mask_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vfunc = np.vectorize(lambda x : 255 if x > 125 else 0)
    img = vfunc(img)
    cv2.imwrite(mask_path,img)

for mask_path in tqdm(mask_paths_list):
    img = cv2.imread(mask_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vfunc = np.vectorize(lambda x : 1 if x ==255 else 0)
    img = vfunc(img)
    cv2.imwrite(mask_path,img)

    


