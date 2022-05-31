import sys # 모델 경로
sys.path.append('../../SkyReplacement')

from SkySegmentation.filter_seg import process_image_or_folder
from SkySegmentation.mmseg_config.utils import increment_jpg_path
import time
import cv2
from Replacement import replace
from mmseg.apis import inference_segmentor, init_segmentor
import numpy as np
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint

from PIL import Image

import mmcv


print('Load Sky Segmentor')
config_file = '/opt/ml/input/final-project-level3-cv-17/SkyReplacement/SkySegmentation/checkpoint/segformer.py'
checkpoint_file = '/opt/ml/input/final-project-level3-cv-17/SkyReplacement/SkySegmentation/checkpoint/epoch_34.pth'
cfg = mmcv.Config.fromfile(config_file)
model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
model.cfg = cfg
model.to('cuda:0')
model.eval()
load_checkpoint(model, checkpoint_file, map_location='cpu')
model.CLASSES =("BackGroud","Sky")

def load_image(img): # bytes to cv2
    encoded_img = np.fromstring(img, dtype = np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def segmentor(dehazed_image, challenge):
    dehazed_image = load_image(dehazed_image)
    print('Sky Segmentation Begin!')
    if challenge:
        # model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
        result = inference_segmentor(model, dehazed_image)
        vfunc = np.vectorize(lambda x : 255 if x ==1 else 0)
        sky_mask = vfunc(result[0]).astype(np.uint8)  

    # filter based
    else:
        sky_mask = process_image_or_folder(dehazed_image)

    sky_mask = Image.fromarray(sky_mask)

    return sky_mask

def replace_sky(img, sky_mask, sky):
    img = load_image(img)
    sky = load_image(sky)
    # mask 이미지는 grayscale
    mask_encoded_img = np.fromstring(sky_mask, dtype = np.uint8)
    sky_mask = cv2.imdecode(mask_encoded_img, cv2.IMREAD_GRAYSCALE)
    print('Sky Replacement Begin!')
    sz = img.shape
     
    # replace the sky
    I_rep = replace.replace_sky(img,sky_mask,sky)
     
    # color transfer
    transfer = replace.color_transfer(sky,sky_mask,I_rep,1)
    mask_edge = cv2.Canny(sky_mask,100,200)
    mask_edge_hwc = cv2.merge([mask_edge, mask_edge, mask_edge])
     
    # guided filtering
    final = replace.guideFilter(img, transfer, mask_edge_hwc, (8,8), 0.01)
    # final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
    final = Image.fromarray(final)
    return final
