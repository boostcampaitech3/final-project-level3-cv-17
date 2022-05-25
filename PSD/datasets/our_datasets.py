import sys
import torch.utils.data as data
import os
from PIL import Image
from random import randrange
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
import torch


class SynTrainData(data.Dataset):
    # RESIDE-OTS
    # BeDDE
    # MRFID
    def __init__(self, crop_size, train_data_dir, haze_folder, gt_folder):
        super().__init__()
        self.data_name = train_data_dir.split('/')[-1]
        self.haze_folder = haze_folder
        self.gt_folder = gt_folder

        self.haze_dir = os.path.join(train_data_dir, self.haze_folder)
        self.gt_dir = os.path.join(train_data_dir, self.gt_folder)
        self.haze_names = list(os.walk(self.haze_dir))[0][2]

        self.crop_size = crop_size

    def __getitem__(self, index):
        crop_width, crop_height =  self.crop_size
        haze_name = self.haze_names[index]
        haze_name = os.path.join(self.haze_dir,haze_name)
        
        gt_name = haze_name.replace(self.haze_folder, self.gt_folder)
        if self.data_name == 'RESIDE-OTS':
            gt_name = gt_name.split('_')[0] + '.jpg'
        elif self.data_name == 'BeDDE':
            gt_name = gt_name.split('_')[0] + '_clear.png'
        elif self.data_name == 'MRFID':
            gt_name = gt_name.split('-')[0] + '0.png'
        

        haze_img = Image.open(haze_name).convert('RGB')
        gt_img = Image.open(gt_name).convert('RGB')

        width, height = haze_img.size

        if width < crop_width or height < crop_height:
            if width < height:
                haze_img = haze_img.resize((260, (int)(height * 260/ width)), Image.ANTIALIAS)
                gt_img = gt_img.resize((260, (int)(height * 260/ width)), Image.ANTIALIAS)
            elif width >= height:
                haze_img = haze_img.resize(((int)(width * 260/ height), 260), Image.ANTIALIAS)
                gt_img = gt_img.resize(((int)(width * 260 / height), 260), Image.ANTIALIAS)
            width, height = haze_img.size
        
        # --- random crop --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])

        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)
        
        if list(haze.shape)[0] != 3 or list(gt.shape)[0] != 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return haze, gt


    def __len__(self):
        return len(self.haze_names)


class RealTrainData_CLAHE(data.Dataset):
    # Hidden
    # RESIDE_RTTS
    def __init__(self, crop_size, train_data_dir, haze_folder, gt_folder):
        super().__init__()
        self.data_name = train_data_dir.split('/')[-1]
        self.haze_folder = haze_folder
        self.gt_folder = gt_folder

        self.haze_dir = os.path.join(train_data_dir, self.haze_folder)
        self.gt_dir = os.path.join(train_data_dir, self.gt_folder)
        self.haze_names = list(os.walk(self.haze_dir))[0][2]

        self.crop_size = crop_size

    def __getitem__(self, index):
        crop_width, crop_height =  self.crop_size
        haze_name = self.haze_names[index]
        haze_name = os.path.join(self.haze_dir,haze_name)

        gt_name = haze_name.replace(self.haze_folder, self.gt_folder)

        haze_img = Image.open(haze_name).convert('RGB')
        gt_img = Image.open(gt_name).convert('RGB')

        width, height = haze_img.size

        if width < crop_width or height < crop_height:
            if width < height:
                haze_img = haze_img.resize((260, int(260 * (height / width))), Image.ANTIALIAS)
                gt_img = gt_img.resize((260, int(260 * (height / width))), Image.ANTIALIAS)
            else:
                haze_img = haze_img.resize((int(260 * (width / height)), 260), Image.ANTIALIAS)
                gt_img = gt_img.resize((int(260 * (width / height)), 260), Image.ANTIALIAS)
            width, height = haze_img.size
        
        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])

        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)

        if list(haze.shape)[0] != 3 or list(gt.shape)[0] != 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return haze, gt


    def __len__(self):
        return len(self.haze_names)


class SynValData(data.Dataset):
    # RESIDE_SOTS_OUT
    def __init__(self, val_data_dir, haze_folder, gt_folder):
        super().__init__()

        self.haze_folder = haze_folder
        self.gt_folder = gt_folder

        self.haze_dir = os.path.join(val_data_dir, self.haze_folder)
        self.gt_dir = os.path.join(val_data_dir, self.gt_folder)
        self.haze_names = list(os.walk(self.haze_dir))[0][2]


    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = haze_name.replace(self.haze_folder, self.gt_folder)
        
        haze_img = Image.open(os.path.join(self.haze_dir,haze_name)).convert('RGB')
        gt_img = Image.open(os.path.join(self.gt_dir,gt_name)).convert('RGB')

        haze_reshaped = haze_img
        haze_reshaped = haze_reshaped.resize((512, 512), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        haze_reshaped = transform_haze(haze_reshaped)
        gt = transform_gt(gt_img)

        if list(haze.shape)[0] != 3 or list(gt.shape)[0] != 3:
            raise Exception('Bad image channel: {}'.format(gt_name))
        
        return haze, haze_reshaped, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)


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
    
    test_data_dir = '/opt/ml/final-project-level3-cv-17/data/RESIDE_SOTS_OUT/hazy'
    test_data_loader = data.DataLoader(ETCDataset(test_data_dir), batch_size=1, shuffle=False, num_workers=8) # For FFA and MSBDN
    for i, data in enumerate(test_data_loader):
       print(f'rst[0] : {data[0].shape}')
       print(f'rst[1] : {data[1].shape}')
       print(f'rst[2] : {data[2]}')