import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision.transforms import Compose, ToTensor

from datasets.our_datasets import ETCDataset
from models.FFA import FFANet
from models.MSBDN import MSBDNNet
from models.dehazeformer import dehazeformer_m
from utils import set_seed

import os
import time
import pyiqa
import argparse
import warnings
warnings.filterwarnings("ignore")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0", help='cuda:0/cuda/cpu')
    parser.add_argument('--backbone', type=str, default='DehazeFormer_m', help='FFA/MSBDN/DehazeFormer_m')
    parser.add_argument('--weight_path', type=str, default='/opt/ml/input/final-project-level3-cv-17/PSD/finetuned_model/Dehazeformer-Finetune.pth')
    parser.add_argument('--data', type=str, default='Hidden')
    parser.add_argument('--min_size', type=int, default=512)
    parser.add_argument('--max_size', type=int, default=2560)
    parser.add_argument('--check_size', type=int, default=16)
    opt = parser.parse_known_args()[0]
    return opt
    
def main(opt):
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device(opt.device)

    if opt.backbone=='FFA' : net = FFANet(3, 19)
    elif opt.backbone=='MSBDN' : net = MSBDNNet()
    elif opt.backbone=='DehazeFormer_m' : net = dehazeformer_m()
    net = nn.DataParallel(net, device_ids=device_ids)
    net.load_state_dict(torch.load(opt.weight_path))
    net.eval()

    metric_NIQE = pyiqa.create_metric('niqe').to(device)
    metric_BRIS = pyiqa.create_metric('brisque').to(device)
    metric_NIMA = pyiqa.create_metric('nima').to(device)

    if opt.device=='cpu': num_workers = 1
    else: num_workers = 8
    
    test_data_dir = f'../data/{opt.data}/hazy/'
    test_data_loader = DataLoader(ETCDataset(test_data_dir, opt.min_size, opt.max_size, opt.check_size),
                                  batch_size=1, shuffle=False, num_workers=num_workers)

    output_dir = 'output/' + opt.weight_path.split('/')[-2] +'/'
    output_dir += opt.weight_path.split('/')[-1].split('.')[0] +'/'+ test_data_dir.split('/')[2] +'/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    NIQE, BRIS, NIMA = 0, 0, 0
    with torch.no_grad():
        for id, test_data in enumerate(test_data_loader):
            # start = time.time()
            haze, haze_A, name = test_data
            haze, haze_A = haze.to(device), haze_A.to(device)
            
            _, pred, T, A, I = net(haze, haze_A, Val=True)
            ts = torch.squeeze(pred.clamp(0, 1).cpu())

            file_name = output_dir + name[0].split('.')[0] + name[0][-4:]
            vutils.save_image(ts, file_name)
            # print(time.time() - start)

            pred = pred.to(device)
            NIQE += metric_NIQE(pred).item()
            BRIS += metric_BRIS(pred).item()
            NIMA += metric_NIMA(pred).item()
        
        print("NIQE", NIQE/len(test_data_loader))
        print("BRIS", BRIS/len(test_data_loader))
        print("NIMA", NIMA/len(test_data_loader))

if __name__=='__main__':
    set_seed(42)
    opt = get_parser()
    main(opt)
