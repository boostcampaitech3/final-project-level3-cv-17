import sys
sys.path.append('../../PSD')

import streamlit as st

import io
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms.functional import to_pil_image

from models.GCA import GCANet
from models.FFA import FFANet
from models.MSBDN import MSBDNNet

import base64

def load_model():
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### MSBDNNet
    PSD = MSBDNNet()
    PSD = nn.DataParallel(PSD, device_ids=device_ids)
    PSD.load_state_dict(torch.load('/opt/ml/input/final-project-level3-cv-17/PSD/pretrained_model/PSD-MSBDN'))
    PSD.eval()

    ### FFANet
    # PSD = FFANet(3, 19)
    # PSD = nn.DataParallel(PSD, device_ids=device_ids)
    # PSD.load_state_dict(torch.load('/opt/ml/input/final-project-level3-cv-17/PSD/pretrained_model/PSD-FFANET'))
    # PSD.eval()

    ### GCANet
    # PSD = GCANet(in_c=4, out_c=3, only_residual=True)
    # PSD = nn.DataParallel(PSD, device_ids=device_ids)
    # PSD.load_state_dict(torch.load('/opt/ml/input/final-project-level3-cv-17/PSD/pretrained_model/PSD-GCANET'))
    # PSD.eval()

    return device, PSD

def get_images(image_bytes: bytes):
    haze_img = Image.open(io.BytesIO(image_bytes))
    haze_img = haze_img.convert("RGB")
    haze_reshaped = haze_img
    haze_reshaped = haze_reshaped.resize((512, 512))

    # --- Transform to tensor --- #
    transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_gt = Compose([ToTensor()])
    haze = transform_haze(haze_img) # torch.Size([3, 3264, 2448])
    haze_reshaped = transform_haze(haze_reshaped) # torch.Size([3, 512, 512])
    
    ### GCANet - edge compute for 4 channels
    # haze_edge = edge_compute(haze)
    # haze = torch.cat((haze, haze_edge), 0)
    # haze_reshaped_edge = edge_compute(haze_reshaped)
    # haze_reshaped = torch.cat((haze_reshaped, haze_reshaped_edge), 0)
    ### 

    haze = torch.unsqueeze(haze, 0)
    haze_reshaped = torch.unsqueeze(haze_reshaped, 0)
    
    return haze, haze_reshaped

def get_prediction(image_bytes: bytes) -> bytes:
    device, PSD = load_model()
    with torch.no_grad():
        haze, haze_A = get_images(image_bytes)  
        haze.to(device)
        print('BEGIN!')

        ### MSBDN & FFA ###
        _, pred, T, A, I = PSD(haze, haze_A, True) # For FFA and MSBDN
        ts = torch.squeeze(pred.clamp(0, 1).cpu())
        ts = to_pil_image(ts)
        # ts = from_image_to_bytes(ts)
        # print(ts)
        # ts = ts.tobytes()
    return ts

        ### GCA ###
    #     pred = PSD(haze, 0, True, False) # For GCA
    #     dehaze = pred.float().round().clamp(0, 255)
    #     out_img = Image.fromarray(dehaze[0].cpu().numpy().astype(np.uint8).transpose(1, 2, 0))

    # return out_img


def edge_compute(x):
    
    x_diffx = torch.abs(x[:,:,1:] - x[:,:,:-1])
    x_diffy = torch.abs(x[:,1:,:] - x[:,:-1,:])

    y = x.new(x.size())
    y.fill_(0)
    y[:,:,1:] += x_diffx
    y[:,:,:-1] += x_diffx
    y[:,1:,:] += x_diffy
    y[:,:-1,:] += x_diffy
    y = torch.sum(y,0,keepdim=True)/3
    y /= 4
    
    return y
