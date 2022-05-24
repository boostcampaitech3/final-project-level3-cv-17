import sys
import torch.utils.data as data
import os
from PIL import Image
from random import randrange
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
import torch

class ETCDataset(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()

        self.haze_dir = val_data_dir
        self.haze_names = list(os.walk(self.haze_dir))[0][2]

    def get_images(self, index):
        haze_name = self.haze_names[index]
        #gt_name = haze_name.split('_')[0] + '.png'
        haze_img = Image.open(os.path.join(self.haze_dir,haze_name)).convert('RGB')
        haze_reshaped = haze_img
        haze_reshaped = haze_reshaped.resize((512, 512))
        #gt_img = Image.open(self.gt_dir + gt_name)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        haze_reshaped = transform_haze(haze_reshaped)
        #haze_edge_data = edge_compute(haze)
        #haze = torch.cat((haze, haze_edge_data), 0)
        #gt = transform_gt(gt_img)

        return haze, haze_reshaped, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)


if __name__=='__main__':
    
    test_data_dir = '../data'
    test_data_loader = data.DataLoader(ETCDataset(test_data_dir), batch_size=1, shuffle=False, num_workers=8) # For FFA and MSBDN
    for i, data in enumerate(test_data_loader):
       print(f'rst[0] : {data[0].shape}')
       print(f'rst[1] : {data[1].shape}')
       print(f'rst[2] : {data[2]}')