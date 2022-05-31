import sys # 모델 경로
sys.path.append('../../PSD')

import warnings # warning 무시
warnings.filterwarnings('ignore')

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

from models.FFA import FFANet
from models.MSBDN import MSBDNNet

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

    return device, PSD

print("Load Dehazing Model")
device, PSD = load_model()

def get_image(image_bytes: bytes):
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

def get_prediction(image_bytes: bytes) -> bytes:
    
    with torch.no_grad():
        haze, haze_A = get_image(image_bytes)  
        haze.to(device)
        print('Hazing Begin!')

        ### MSBDN & FFA ###
        _, pred, T, A, I = PSD(haze, haze_A, True) # For FFA and MSBDN
        ts = torch.squeeze(pred.clamp(0, 1).cpu())
        ts = to_pil_image(ts)
    return ts
