import warnings # warning 무시
warnings.filterwarnings('ignore')

import io
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms.functional import to_pil_image

from .models.Dehazing.FFA import FFANet
from .models.Dehazing.MSBDN import MSBDNNet
from .models.Dehazing.Dehazeformer import dehazeformer_m

def load_model():
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = 'Dehazeformer' # FFA / MSBDN / Dehazeformer
    data = 'Hidden' # Crawling / Hidden

    if backbone=='FFA' : net = FFANet(3, 19)
    elif backbone=='MSBDN' : net = MSBDNNet()
    elif backbone=='Dehazeformer' : net = dehazeformer_m()
    net = nn.DataParallel(net, device_ids=device_ids)

    if backbone=='FFA' : model_path = 'app/weights/Dehazing/PSD-FFANET'
    elif backbone=='MSBDN' : model_path = 'app/weights/Dehazing/PSD-MSBDN'
    elif backbone=='Dehazeformer' : model_path = 'app/weights/Dehazing/Dehazeformer-Finetune.pth' # PSD-Dehazeformer
    net.load_state_dict(torch.load(model_path))
    net.eval()

    return device, net

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

        ### FFA / MSBDN / Dehazeformer ###
        _, pred, T, A, I = PSD(haze, haze_A, True) 
        ts = torch.squeeze(pred.clamp(0, 1).cpu())
        ts = to_pil_image(ts)
    return ts
