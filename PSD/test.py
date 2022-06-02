import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from datasets.our_datasets import ETCDataset
from models.FFA import FFANet
from models.MSBDN import MSBDNNet
from models.dehazeformer import dehazeformer_m

import os
import warnings
warnings.filterwarnings("ignore")


device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

backbone = 'Dehazeformer' # FFA / MSBDN / Dehazeformer
data = 'Hidden' # Crawling / Hidden

if backbone=='FFA' : net = FFANet(3, 19)
elif backbone=='MSBDN' : net = MSBDNNet()
elif backbone=='Dehazeformer' : net = dehazeformer_m()
net = nn.DataParallel(net, device_ids=device_ids)

if backbone=='FFA' : model_path = 'pretrained_model/PSD-FFANET'
elif backbone=='MSBDN' : model_path = 'pretrained_model/PSD-MSBDN'
elif backbone=='Dehazeformer' : model_path = 'pretrained_model/PSD-Dehazeformer.pth'
net.load_state_dict(torch.load(model_path))
net.eval()

test_data_dir = f'../data/{data}/hazy/'
test_data_loader = DataLoader(ETCDataset(test_data_dir, backbone=backbone), batch_size=1, shuffle=False, num_workers=8) # For FFA and MSBDN

output_dir = 'output/' + model_path.split('/')[-2] +'/'+ model_path.split('/')[-1].split('.')[0] +'/'+ test_data_dir.split('/')[2] +'/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for _, test_data in enumerate(test_data_loader):
        haze, haze_A, name = test_data
        haze, haze_A = haze.to(device), haze_A.to(device)
        
        if backbone in ['MSBDN','Dehazeformer']:
            width, height = haze.size()[2], haze.size()[3]
            if width % 16 != 0 or height % 16 != 0:
                haze = F.upsample(haze, [width + 16 - width % 16, height + 16 - height % 16], mode='bilinear')
        
        _, pred, T, A, I = net(haze, haze_A, Val=True)
        ts = torch.squeeze(pred.clamp(0, 1).cpu())

        file_name = output_dir + name[0].split('.')[0] + name[0][-4:]
        vutils.save_image(ts, file_name)
