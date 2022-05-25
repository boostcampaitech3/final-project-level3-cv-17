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

def load_model():
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PSD = MSBDNNet()
    PSD = nn.DataParallel(PSD, device_ids=device_ids)
    PSD.load_state_dict(torch.load('/opt/ml/input/final-project-level3-cv-17/PSD/pretrained_model/PSD-MSBDN'))
    PSD.eval()

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
    haze = torch.unsqueeze(haze, 0)
    haze_reshaped = torch.unsqueeze(haze_reshaped, 0)
    
    return haze, haze_reshaped

@st.cache
def get_prediction(image_bytes: bytes) -> bytes:
    device, PSD = load_model()
    with torch.no_grad():
        haze, haze_A = get_images(image_bytes)  
        haze.to(device)
        print('BEGIN!')
        # pred = net(haze, 0, True, False) # For GCA
        _, pred, T, A, I = PSD(haze, haze_A, True) # For FFA and MSBDN
        
        ### GCA ###
        # dehaze = pred.float().round().clamp(0, 255)
        # out_img = Image.fromarray(dehaze[0].cpu().numpy().astype(np.uint8).transpose(1, 2, 0))
        # out_img.save(output_dir + name[0].split('.')[0] + '_MyModel_{}.png'.format(batch_id))
        ###########
        ### FFA & MSBDN ###
        ts = torch.squeeze(pred.clamp(0, 1).cpu())
        ###################
        ts = to_pil_image(ts)
    return ts
