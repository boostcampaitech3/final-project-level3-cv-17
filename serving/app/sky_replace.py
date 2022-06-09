import time
import cv2
import numpy as np
import imutils
from PIL import Image

from .models.Sky import replace
from .models.Sky import skyselect as select
from .models.Sky.filter_seg import find_hsv_upper_lower_threshold_for_skyarea_and_weather_condition
from .models.Sky.clear_edge import edge_mask,variance_filter

from mmseg.apis import inference_segmentor
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
import mmcv

from skimage.exposure import match_histograms # for histogram matching

SKY_IMAGES_PATH = '/opt/ml/input/final-project-level3-cv-17/data/sky_images/database/img/*'

def load_model():
    config_file = 'app/models/Sky/segformer.py' 
    checkpoint_file = 'app/weights/Sky/SkySegmentation/seg_epoch_20.pth'
    cfg = mmcv.Config.fromfile(config_file)
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.cfg = cfg
    model.to('cuda:0')
    model.eval()
    load_checkpoint(model, checkpoint_file, map_location='cpu')
    model.CLASSES =("BackGroud","Sky")
    return model

print('Load Sky Segmentor')
model = load_model()

def segment_model_prediction(img):
    result = inference_segmentor(model, img)
    sky_mask = result[0].astype(np.uint8)  
    return sky_mask

def blue_channel_filter(img):
    img_b = img[:,:,2] #clahe.apply() 
    img_b = cv2.threshold(img_b.astype(np.uint8),0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_b = img_b[1]
    return img_b

def low_variance_filter(img, sky_mask):
    img_var = variance_filter(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), window_size=3)
    varianceBasedSkyMask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    _,_, weather_condition  = find_hsv_upper_lower_threshold_for_skyarea_and_weather_condition(sky_mask, hls_img)
    varianceThreshold = 10
    if weather_condition == "day":
        varianceThreshold = 10
    if weather_condition == "dayCloudy":
        varianceThreshold = 150 # threshold is increased for cloudy images, as a low threshold will make generate outline by the cloud
    if weather_condition == "night":
        varianceThreshold = 5 # threshold is decreased, as the variance in night image for sky is low. This could avoid missing sky pixel
    if weather_condition == "nightCloudy":
        varianceThreshold = 20

    varianceBasedSkyMask[img_var <= varianceThreshold] = 1 # 1

    # edge post processing
    temp_edge = edge_mask(varianceBasedSkyMask)
    varianceBasedSkyMask = cv2.bitwise_or(temp_edge,varianceBasedSkyMask)
    # varianceBasedSkyMask -=temp_edge
    # varianceBasedSkyMask = np.clip(varianceBasedSkyMask,0,1)
    return varianceBasedSkyMask

def load_image(img): # bytes to cv2
    encoded_img = np.fromstring(img, dtype = np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def image_resize(img):
    height, width = img.shape[:2]
    if height >= width:
        img = imutils.resize(img,height=1280) 
    else:
        img = imutils.resize(img,width=1280) 
    return img

def load_mask_image(mask):
    mask = np.fromstring(mask, dtype = np.uint8)
    mask = cv2.imdecode(mask, cv2.IMREAD_GRAYSCALE)
    return mask


def segmentor(img):
    # global dehazed_image, sky_mask
    img = image_resize(load_image(img))
    print('Sky Segmentation Begin!')

    # segmentation model prediction
    sky_mask = segment_model_prediction(img)

    # blue channel filter
    img_b = blue_channel_filter(img)

    # low variance filter
    varianceBasedSkyMask = low_variance_filter(img, sky_mask)

    sky_mask = cv2.bitwise_and(sky_mask, varianceBasedSkyMask)
    sky_mask = cv2.bitwise_and(sky_mask, img_b)

    ## Mask Post Processing - Closing & Opening 
    # squareSe = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) # 3x3 RECT kernel

    # Perform opening to remove small noise in fine_tuned_sky_mask
    # sky_mask = cv2.morphologyEx(sky_mask,cv2.MORPH_OPEN,squareSe, iterations=3)

    # Perform closing on non-sky area to bridge small gap between non-sky area in fine_tuned_sky_mask
    # we are performing closing on non-sky area, hence we invert the sky mask
    # skymask_closing_non_roi = cv2.morphologyEx(1 - skymask_opening,cv2.MORPH_CLOSE,squareSe, iterations=1) 

    # # After closing, invert the inverted mask.
    # sky_mask = 1 - skymask_closing_non_roi
    # sky_mask = 1 - sky_mask

    vfunc = np.vectorize(lambda x : 255 if x ==1 else 0)
    sky_mask = vfunc(sky_mask).astype(np.uint8)    

    sky_mask = Image.fromarray(sky_mask)
    return sky_mask


def select_sky_paths(dehazed_image, sky_mask, option):
    dehazed_image = image_resize(load_image(dehazed_image))
    sky_mask = load_mask_image(sky_mask)

    sky_paths = select.select_sky(dehazed_image, sky_mask, option=option) 
    
    # for test, select best clip score image
    # sky_path = sky_paths[0]
    return sky_paths

def check_sky(sky_mask, ref_mask):
    sky_mask = load_mask_image(sky_mask)
    ref_mask = load_mask_image(ref_mask)
    state = 'OK' # replace 결과 및 상태 확인
    # direct sky_path
    if select.determine(ref_mask, sky_mask) and select.mask_intersection(sky_mask, ref_mask):
        pass
    else:
        print('넣어주신 하늘 이미지를 사용할 경우 합성시 왜곡이 발생할 수 있습니다.')
        state = 'sky_image_dismatch'
    # Exception
    if np.all((sky_mask == 0)):
        print('이미지에 하늘이 존재하지 않습니다.')
        state = 'input_image_no_sky'
    return state

def replace_sky(dehazed_image, sky_mask, sky, ref_mask):
    print('Sky Replacement Begin!')
    dehazed_image = image_resize(load_image(dehazed_image))
    sky = load_image(sky)
    sky_mask = load_mask_image(sky_mask)
    ref_mask = load_mask_image(ref_mask)

    # 1) color transfer
    I_rep = replace.replace_sky(dehazed_image, sky_mask, sky) # replace the sky
    transfer = replace.color_transfer(sky,sky_mask,I_rep,1) # color transfer

    # 2) histogram matching 
    # g = 1.5 # histogram matchong 전에 하늘 대비, 밝기 낮추기?
    # sky_float = sky.astype(np.float)
    # sky_color_down = ((sky_float / 255) ** (1 / g)) * 255
    # sky_color_down = sky_color_down.astype(np.uint8)
    # sky_color_down = 0.5 * sky + 75
    # sky_color_down = np.clip(sky_color_down, 0, 255).astype(np.uint8)
    # matched_dehazed_image = match_histograms(dehazed_image, sky_color_down, channel_axis=-1) # histogram matching
    # I_rep = replace.replace_sky(matched_dehazed_image, sky_mask, sky) # replace the sky

    # 그라데이션
    mask_grad = np.array(Image.open('/opt/ml/input/final-project-level3-cv-17/data/gray_gradient/gray_gradient_v.jpg').resize(dehazed_image.shape[1::-1], Image.BILINEAR))
    mask_grad = mask_grad / 255
    final = dehazed_image * mask_grad + transfer * (1 - mask_grad)  # 1) transfer / 2) I_rep
    final = np.clip(final, 0, 255).astype(np.uint8)
    
    final = Image.fromarray(final)
    return final
