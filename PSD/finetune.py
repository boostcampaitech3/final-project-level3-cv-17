import os
import argparse
import time
import copy
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import torchvision.utils as vutils

from datasets.concat_dataset import ConcatDataset
from datasets.our_datasets import SynTrainData, RealTrainData_CLAHE, SynValData
from utils import *
from losses.energy_functions import *
from losses.loss_functions import *
from wandb_setup import wandb_login, wandb_init

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--backbone', type=str, default='MSBDNNet', help='Backbone model(GCANet/FFANet/MSBDNNet)')

parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--train_batch_size', type=int, default=6)
parser.add_argument('--val_batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--crop_size', type=int, default=256, help='size of random crop')
parser.add_argument('--category', type=str, default='outdoor', help='dataset type: indoor / outdoor') # outdoor only
parser.add_argument('--print_freq', type=int, default=20)

parser.add_argument('--work_dir', type=str, default='/opt/ml/final-project-level3-cv-17/PSD/work_dirs')
parser.add_argument('--label_dir', type=str, default='/opt/ml/final-project-level3-cv-17/data/BeDDE')
parser.add_argument('--unlabel_dir', type=str, default='/opt/ml/final-project-level3-cv-17/data/RESIDE_RTTS')
parser.add_argument('--pseudo_gt_dir', type=str, default='/data/nnice1216/Dehazing/') # not use now
parser.add_argument('--val_dir', type=str, default='/opt/ml/final-project-level3-cv-17/data/RESIDE_SOTS_OUT')
parser.add_argument('--pretrain_model_dir', type=str, default='/opt/ml/final-project-level3-cv-17/PSD/pretrained_model')

parser.add_argument('--lambda_dc', type=float, default=2e-3)
parser.add_argument('--lambda_bc', type=float, default=3e-2)
parser.add_argument('--lambda_CLAHE', type=float, default=1)
parser.add_argument('--lambda_rec', type=float, default=1)
parser.add_argument('--lambda_lwf_label', type=float, default=1)
parser.add_argument('--lambda_lwf_unlabel', type=float, default=1)
parser.add_argument('--lambda_lwf_sky', type=float, default=1)

opt = parser.parse_known_args()[0]
work_dir_exp = increment_path(os.path.join(opt.work_dir, 'exp'))
make_directory(work_dir_exp)

wandb_login()
wandb_init(opt, work_dir_exp)

set_seed(42)

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

crop_size = [opt.crop_size, opt.crop_size]


net = load_model(opt.backbone, opt.pretrain_model_dir, device, device_ids)

net_o = copy.deepcopy(net)
net_o.eval()

optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)


print('Begin DataLoader!')

syn_dataset_name = opt.label_dir.split('/')[-1]
if syn_dataset_name == 'RESIDE-OTS':
    syn_dataset = SynTrainData(crop_size, opt.label_dir, 'hazy/part1', 'gt')
else:
    syn_dataset = SynTrainData(crop_size, opt.label_dir, 'hazy', 'gt')

train_data_loader = DataLoader(
                ConcatDataset(
                    syn_dataset,
                    RealTrainData_CLAHE(crop_size, opt.unlabel_dir, 'hazy', 'gt_clahe')
                ),
                batch_size=opt.train_batch_size,
                shuffle=False,
                num_workers=opt.num_workers,
                drop_last=True)

val_data_loader = DataLoader(
                SynValData(opt.val_dir, 'hazy', 'gt'),
                batch_size=opt.val_batch_size,
                shuffle=False,
                num_workers=opt.num_workers)

print('Dataloader Done')

train_psnr = 0

loss_dc = energy_dc_loss()

for epoch in range(opt.num_epochs):
    psnr_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch, category=opt.category) # indoor: decay_step=20 # outdoor: decay_step=10
 
    for batch_id, (label_train_data, unlabel_train_data) in enumerate(train_data_loader):
        
        # --- load data --- #
        label_haze, label_gt = label_train_data
        unlabel_haze, unlabel_gt = unlabel_train_data
        
        label_haze = label_haze.to(device)
        label_gt = label_gt.to(device)
        unlabel_haze = unlabel_haze.to(device)
        unlabel_gt = unlabel_gt.to(device)

        # --- train --- #
        optimizer.zero_grad()
        net.train()
        
        out_label, J_label, T_label, _, _ = net(label_haze)
        out_label_o, J_label_o, T_label_o, _, _ = net_o(label_haze)
        
        out, J, T, A, I = net(unlabel_haze)
        out_o, J_o, T_o, _, _ = net_o(unlabel_haze)
        I2 = T * unlabel_gt + (1 - T) * A

        # --- losses --- #
        energy_dc_loss = loss_dc(unlabel_haze, T)
        bc_loss = bright_channel(unlabel_haze, T)
        CLAHE_loss = F.smooth_l1_loss(I2, unlabel_haze)
        rec_loss = F.smooth_l1_loss(I, unlabel_haze)

        lwf_loss_sky = lwf_sky(unlabel_haze, J, J_o)
        lwf_loss_label = F.smooth_l1_loss(out_label, out_label_o)
        lwf_loss_unlabel = F.smooth_l1_loss(out, out_o)

        loss = opt.lambda_dc*energy_dc_loss + opt.lambda_bc*bc_loss + opt.lambda_CLAHE*CLAHE_loss
        loss += opt.lambda_rec*rec_loss + opt.lambda_lwf_sky*lwf_loss_sky
        loss += opt.lambda_lwf_label*lwf_loss_label + opt.lambda_lwf_unlabel*lwf_loss_unlabel

        loss.backward()
        optimizer.step()

        if not (batch_id % opt.print_freq):
            print('Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, batch_id, loss))


    # --- save model --- #
    save_path = os.path.join(work_dir_exp, f'Epoch{epoch}.pth')
    torch.save(net.state_dict(), save_path)

    # --- Use the evaluation model in testing --- #
    net.eval()
    
    val_psnr, val_ssim = validation(net, opt.backbone, val_data_loader, device, opt.category)
    one_epoch_time = time.time() - start_time
    print_log(epoch+1, opt.num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, opt.category)

# --- output test images --- #
generate_test_images(net, TestData, opt.num_epochs, (0, opt.num_epochs - 1))