from SkySegmentation.filter_seg import find_hsv_upper_lower_threshold_for_skyarea_and_weather_condition
from SkySegmentation.clear_edge import edge_mask,variance_filter
from SkySegmentation.mmseg_config.utils import increment_jpg_path
import argparse
import time
import time
import cv2
import sys
sys.path.append('/opt/ml/input/final-project-level3-cv-17/SkyReplacement/Replacement/SkySelection')
from Replacement import replace
from Replacement.SkySelection import skyselect as select
from mmseg.apis import inference_segmentor, init_segmentor
import numpy as np
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
import mmcv
import imutils
from pykuwahara import kuwahara
from skimage.exposure import match_histograms

SKY_IMAGES_PATH = '/opt/ml/input/final-project-level3-cv-17/data/sky_images/database/img/*'


def model_prediction(img):
    config_file = '/opt/ml/input/final-project-level3-cv-17/SkyReplacement/SkySegmentation/checkpoint/segformer_3.py'
    checkpoint_file ='/opt/ml/input/final-project-level3-cv-17/SkyReplacement/SkySegmentation/mmseg_config/work_dirs/exp89/epoch_80.pth' #'/opt/ml/input/final-project-level3-cv-17/SkyReplacement/SkySegmentation/checkpoint/epoch_20.pth'
    cfg = mmcv.Config.fromfile(config_file)
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.cfg = cfg
    model.to('cuda:0')
    model.eval()
    load_checkpoint(model, checkpoint_file, map_location='cpu')
    model.CLASSES =("BackGroud","Sky")
    result = inference_segmentor(model, img)
    sky_mask = result[0].astype(np.uint8)  
    return sky_mask

def blue_channel_filter(img):
    img_b = img[:,:,2]#clahe.apply() 
    img_b = cv2.threshold(img_b.astype(np.uint8),0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_b = img_b[1]
    return img_b

def low_variance_filter(img,sky_mask):
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

if __name__ =="__main__":
    parser = argparse.ArgumentParser(
        description='Parser for sky segmentation script')
        
    parser.add_argument('--image_path', type=str,
                                help='path to a dehazed image or folder of images', required=True)
    parser.add_argument('--sky_path', type=str,
                                help='path to a sky image')
    parser.add_argument('--challenge', dest = 'challenge',  action='store_true', help = 'If now challenge case, do model ')
    parser.set_defaults(challenge=True)
    parser.add_argument('--check', dest = 'check',  action='store_true', help = 'If now challenge case, do model ')
    parser.set_defaults(check=False)
    parser.add_argument('--option', type=str,
                                help='user select optoin', required=True)
    args =  parser.parse_args()
    start_time = time.time()  

    img=cv2.cvtColor(cv2.imread(args.image_path,1), cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    if height >= width:
        img = imutils.resize(img,height=1280) 
    else:
        img = imutils.resize(img,width=1280) 

    print('Start')

    if args.challenge :

        # segmentation model prediction
        sky_mask = model_prediction(img)

        # blue channel filter
        img_b = blue_channel_filter(img)

        # low variance filter
        varianceBasedSkyMask = low_variance_filter(img,sky_mask)


        sky_mask = cv2.bitwise_and(sky_mask,varianceBasedSkyMask)
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




    # select the sky
    if args.sky_path:
        sky_path = args.sky_path 
    else:
        sky_paths = select.select_sky(args.image_path, sky_mask, option= args.option) 

    # for test, select best clip score image
    sky_path = sky_paths[0]
    print(sky_path)

    sky=cv2.cvtColor(cv2.imread(sky_path), cv2.COLOR_BGR2RGB)
    mask_path = sky_path.replace('img','mask')
    ref_mask = cv2.imread(mask_path,0) 
    

    # direct sky_path
    if select.determine(ref_mask, sky_mask) and select.mask_intersection(sky_mask, ref_mask):
        pass
    else:
        print('넣어주신 하늘 이미지를 사용할 경우 합성시 왜곡이 발생할 수 있습니다.')


    # replace the sky
    I_rep=replace.replace_sky(img,sky_mask,sky,ref_mask)


    # color transfer
    transfer = replace.color_transfer(sky,sky_mask,I_rep,1)
    final = cv2.cvtColor(transfer, cv2.COLOR_RGB2BGR)
    

    # save result file
    reusult_path = increment_jpg_path('/opt/ml/input/final-project-level3-cv-17/data/sky_replacement/','result')
    cv2.imwrite(reusult_path,final)
    

    if args.check:
        original_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mask_bw = cv2.cvtColor(sky_mask, cv2.COLOR_RGB2BGR)
        sky = cv2.cvtColor(sky, cv2.COLOR_RGB2BGR)
        I_rep = cv2.cvtColor(I_rep, cv2.COLOR_RGB2BGR)
        transfer = cv2.cvtColor(transfer, cv2.COLOR_RGB2BGR)
        cv2.imwrite('/opt/ml/input/final-project-level3-cv-17/data/sky_replacement_check/1.png',original_img)
        cv2.imwrite('/opt/ml/input/final-project-level3-cv-17/data/sky_replacement_check/2.png',sky_mask)
        cv2.imwrite('/opt/ml/input/final-project-level3-cv-17/data/sky_replacement_check/3.png',sky)
        cv2.imwrite('/opt/ml/input/final-project-level3-cv-17/data/sky_replacement_check/4.png',I_rep)
        cv2.imwrite('/opt/ml/input/final-project-level3-cv-17/data/sky_replacement_check/5.png',transfer)
        cv2.imwrite('/opt/ml/input/final-project-level3-cv-17/data/sky_replacement_check/6.png',final)

    print("Total Processing time: {} seconds".format(time.time() - start_time))


