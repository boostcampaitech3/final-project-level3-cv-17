import time
import cv2
import numpy as np
from .models import replace
from .models import skyselect as select
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
import mmcv
import imutils
from pykuwahara import kuwahara
from PIL import Image
from skimage.exposure import match_histograms

SKY_IMAGES_PATH = '/opt/ml/input/final-project-level3-cv-17/data/sky_images/database/img/*'

def load_model():
    print('Load Sky Segmentor')
    config_file = 'app/models/segformer.py'
    checkpoint_file = 'app/weights/epoch_20.pth'
    cfg = mmcv.Config.fromfile(config_file)
    # model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.cfg = cfg
    model.to('cuda:0')
    model.eval()
    load_checkpoint(model, checkpoint_file, map_location='cpu')
    model.CLASSES =("BackGroud","Sky")
    return model

model = load_model()

# Global 이미지
# dehazed_image = ''
# sky_mask = ''

def load_image_resize(img): # bytes to cv2
    encoded_img = np.fromstring(img, dtype = np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    if height >= width:
        img = imutils.resize(img,height=1280) 
    else:
        img = imutils.resize(img,width=1280) 
    return img

def load_image(img):
    encoded_img = np.fromstring(img, dtype = np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def variance_filter(img_gray, window_size = 3): # osth treshold랑 비슷한 느낌인데
    """
    Variance filter
    Calculate the variance in the sliding window
    """
    img_gray = img_gray.astype(np.float64)
    # img_gray = cv2.Laplacian(img_gray,cv2.CV_8U,ksize=3)
    # (cv2.sqrBoxFilter(x,-1,(window_size, window_size))for x in (img_gray, img_gray*img_gray))
    # Variance calculate using vectorized operation. As using a 2d loop to slide the window is too time consuming.
    wmean, wsqrmean = (cv2.boxFilter(x, -1, (window_size, window_size), borderType=cv2.BORDER_REPLICATE) for x in (img_gray, img_gray*img_gray))
    return wsqrmean - wmean*wmean

def segmentor(img, challenge):
    # global dehazed_image, sky_mask
    img = load_image_resize(img)
    print('Sky Segmentation Begin!')

    result = inference_segmentor(model, img)
    sky_mask = result[0].astype(np.uint8)  

    # sky_mask = vfunc(result[0]).astype(np.uint8)  
    img_kuwahara = kuwahara(img, method='mean', radius=1)
    # img_kuwahara = img
    img_b = img_kuwahara[:,:,2]#clahe.apply() 
    # vfunc = np.vectorize(lambda x : 255 if x >210 else 0)   
    img_b = cv2.threshold(img_b.astype(np.uint8),0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_b = img_b[1]
    # cv2.imwrite('./b_mask.png',img_b)
    # sky_mask_2 = process_image_or_folder(args)    
    # cv2.imwrite('./2_mask.png',sky_mask_2)

    img_var = variance_filter(cv2.cvtColor(img_kuwahara, cv2.COLOR_BGR2GRAY), window_size=3)
    varianceBasedSkyMask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    varianceBasedSkyMask[img_var <= 150] = 1 # 1
    # cv2.imwrite('./var_mask.png',varianceBasedSkyMask)

    # sky_mask = cv2.bitwise_and(sky_mask,sky_mask_2)

    sky_mask = cv2.bitwise_and(sky_mask,varianceBasedSkyMask)
    sky_mask = cv2.bitwise_and(sky_mask, img_b)
    # squareSe = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)) # 3x3 RECT kernel

    # sky_mask = cv2.erode(1-sky_mask, squareSe,iterations=3)#, borderType, borderValue)
    # Step 9: Perform opening to remove small noise in fine_tuned_sky_mask
    # skymask_opening = cv2.morphologyEx(1-sky_mask,cv2.MORPH_OPEN,squareSe, iterations=3)
    # # Step 10: Perform closing on non-sky area to bridge small gap between non-sky area in fine_tuned_sky_mask
    # # we are performing closing on non-sky area, hence we invert the sky mask
    # skymask_closing_non_roi = cv2.morphologyEx(1 - sky_mask,cv2.MORPH_CLOSE,squareSe, iterations=1) 
    
    # # After closing, invert the inverted mask.
    # sky_mask = 1 - skymask_closing_non_roi
    # sky_mask = 1 - sky_mask
    vfunc = np.vectorize(lambda x : 255 if x ==1 else 0)
    sky_mask = vfunc(sky_mask).astype(np.uint8)  
    # cv2.imwrite('./mask.png',sky_mask)

    sky_mask = Image.fromarray(sky_mask)
    return sky_mask

def select_sky_paths(dehazed_image, sky_mask, option):
    # global dehazed_image, sky_mask
    # if sky_path:
    #     sky_path = sky_path 
    # else: # 애초에 가능한 이미지들만 보여주자.
    dehazed_image = load_image_resize(dehazed_image)
    # mask 이미지는 grayscale
    sky_mask = np.fromstring(sky_mask, dtype = np.uint8)
    sky_mask = cv2.imdecode(sky_mask, cv2.IMREAD_GRAYSCALE)

    sky_paths = select.select_sky(dehazed_image, sky_mask, option=option) 
    
    # for test, select best clip score image
    # sky_path = sky_paths[0]

    return sky_paths


def replace_sky(dehazed_image, sky_mask, sky):
    # global dehazed_image, sky_mask
    dehazed_image = load_image_resize(dehazed_image)
    sky = load_image(sky)
    # mask 이미지는 grayscale
    mask_encoded_img = np.fromstring(sky_mask, dtype = np.uint8)
    sky_mask = cv2.imdecode(mask_encoded_img, cv2.IMREAD_GRAYSCALE)
    print('Sky Replacement Begin!')
    sz = dehazed_image.shape

    # 1) color transfer
    # replace the sky
    # I_rep = replace.replace_sky(dehazed_image, sky_mask, sky)

    # color transfer
    # transfer = replace.color_transfer(sky,sky_mask,I_rep,1)
    
    # final = Image.fromarray(transfer)

    # 2) histogram matching
    matched_dehazed_image = match_histograms(dehazed_image, sky, channel_axis=-1)
    matched_dehazed_image = np.clip(matched_dehazed_image, 0, 255).astype(np.uint8)

    # # replace the sky
    I_rep = replace.replace_sky(matched_dehazed_image, sky_mask, sky)

    mask_grad = np.array(Image.open('/opt/ml/input/final-project-level3-cv-17/data/gray_gradient_v7.jpg').resize(dehazed_image.shape[1::-1], Image.BILINEAR))
    mask_grad = mask_grad / 255
    final = dehazed_image * mask_grad + I_rep * (1 - mask_grad)  # transfer, I_rep
    final = np.clip(final, 0, 255).astype(np.uint8)
    final = Image.fromarray(final)

    # mask_edge=cv2.Canny(sky_mask,100,200)
    # mask_edge_hwc=cv2.merge([mask_edge, mask_edge, mask_edge])
    # guided filtering
    # final =replace.guideFilter(img,transfer,mask_edge_hwc,(3,3),0.00000001)
    # final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
    return final
