import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from datasets.our_datasets import ETCDataset
from datasets.pretrain_datasets import TestData_GCA
from models.GCA import GCANet
from models.FFA import FFANet
from models.MSBDN import MSBDNNet
from utils import make_directory

import warnings
warnings.filterwarnings("ignore")


test_data_dir = '../data/Hidden/hazy/'
# test_data_dir = '../data/sample/Hidden/'
    
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# net = GCANet(in_c=4, out_c=3, only_residual=True).to(device)
# net = FFANet(3, 19)
net = MSBDNNet()

net = nn.DataParallel(net, device_ids=device_ids)

# net.load_state_dict(torch.load('PSD-GCANET'))
# net.load_state_dict(torch.load('PSD-FFANET'))
net.load_state_dict(torch.load('/opt/ml/final-project-level3-cv-17/PSD/pretrained_model/PSD-MSBDN'))

net.eval()

# test_data_loader = DataLoader(TestData_GCA(test_data_dir), batch_size=1, shuffle=False, num_workers=8) # For GCA
test_data_loader = DataLoader(ETCDataset(test_data_dir), batch_size=1, shuffle=False, num_workers=8) # For FFA and MSBDN


output_dir = '/opt/ml/final-project-level3-cv-17/PSD/output/' + test_data_dir.split('/')[2] + '/'
make_directory(output_dir)
    
with torch.no_grad():
    for batch_id, val_data in enumerate(test_data_loader):
        # haze, name = val_data # For GCA
        haze, haze_A, name = val_data # For FFA and MSBDN
        haze.to(device)

        # vutils.save_image(haze_A, output_dir + name[0].split('.')[0] + '_MyModel_{}.png'.format(batch_id))

        # pred = net(haze, 0, True, False) # For GCA
        _, pred, T, A, I = net(haze, haze_A, True) # For FFA and MSBDN
        
        ### GCA ###
        # dehaze = pred.float().round().clamp(0, 255)
        # out_img = Image.fromarray(dehaze[0].cpu().numpy().astype(np.uint8).transpose(1, 2, 0))
        # out_img.save(output_dir + name[0].split('.')[0] + '_MyModel_{}.png'.format(batch_id))
        ###########
        ### FFA & MSBDN ###
        ts = torch.squeeze(pred.clamp(0, 1).cpu())
        vutils.save_image(ts, output_dir + name[0].split('.')[0] + '_MyModel_{}.png'.format(batch_id))
        ###################
