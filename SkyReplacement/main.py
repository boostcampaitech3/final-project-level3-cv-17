from SkySegmentation.filter_seg import process_image_or_folder
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

SKY_IMAGES_PATH = '/opt/ml/input/final-project-level3-cv-17/data/sky_images/database/img/*'


if __name__ =="__main__":
    parser = argparse.ArgumentParser(
        description='Parser for sky segmentation script')
        
    parser.add_argument('--image_path', type=str,
                                help='path to a dehazed image or folder of images', required=True)
    parser.add_argument('--sky_path', type=str,
                                help='path to a sky image')
    parser.add_argument('--challenge', dest = 'challenge',  action='store_true', help = 'If now challenge case, do model ')
    parser.set_defaults(challenge=False)
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
        config_file = '/opt/ml/input/final-project-level3-cv-17/SkyReplacement/SkySegmentation/checkpoint/segformer.py'
        checkpoint_file = '/opt/ml/input/final-project-level3-cv-17/SkyReplacement/SkySegmentation/mmseg_config/work_dirs/exp82/epoch_20.pth' #'/opt/ml/input/final-project-level3-cv-17/SkyReplacement/SkySegmentation/checkpoint/epoch_20.pth'
        cfg = mmcv.Config.fromfile(config_file)
        # model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        model.cfg = cfg
        model.to('cuda:0')
        model.eval()
        load_checkpoint(model, checkpoint_file, map_location='cpu')
        model.CLASSES =("BackGroud","Sky")
        result = inference_segmentor(model, img)
        vfunc = np.vectorize(lambda x : 255 if x ==1 else 0)
        sky_mask = vfunc(result[0]).astype(np.uint8)  
        img_b_clahe = img[:,:,2]#clahe.apply() 
        vfunc = np.vectorize(lambda x : 255 if x >210 else 0)   
        img_b = vfunc(img_b_clahe).astype('uint8')

        sky_mask = cv2.bitwise_and(sky_mask,img_b)

    # filter based
    else:
        sky_mask = process_image_or_folder(args) # 이게 resize처리가 안되는 중
    
    sz=img.shape

    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) #CLAHE 생성
    # img_b_clahe = img[:,:,2]#clahe.apply() 
    # vfunc = np.vectorize(lambda x : 255 if x >210 else 0)   
    # img_b = vfunc(img_b_clahe).astype('uint8')

    # sky_mask = cv2.bitwise_and(sky_mask,img_b)

    # select the sky
    if args.sky_path:
        sky_path = args.sky_path # 애초에 가능한 이미지들만 보여주자.
    else:
        sky_paths = select.select_sky(args.image_path, sky_mask, option= args.option) 

    # for test, select best clip score image
    sky_path = sky_paths[0]
    print(sky_path)

    sky=cv2.cvtColor(cv2.imread(sky_path), cv2.COLOR_BGR2RGB)
    mask_path = sky_path.replace('img','mask')
    ref_mask = cv2.imread(mask_path,0) # 사용할 하늘 이미지들에 대해서는 미리 마스크가 추출되어 있어야함.


    # replace the sky
    I_rep=replace.replace_sky(img,sky_mask,sky,ref_mask)

    # color transfer
    transfer = replace.color_transfer(sky,sky_mask,I_rep,1)
    mask_edge=cv2.Canny(sky_mask,100,200)
    mask_edge_hwc=cv2.merge([mask_edge, mask_edge, mask_edge])

    # guided filtering
    final =replace.guideFilter(img,transfer,mask_edge_hwc,(8,8),0.01)

    final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)

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

