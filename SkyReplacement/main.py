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
from pykuwahara import kuwahara


SKY_IMAGES_PATH = '/opt/ml/input/final-project-level3-cv-17/data/sky_images/database/img/*'
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

    
    sz=img.shape

    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) #CLAHE 생성
    # img_b_clahe = img[:,:,2]#clahe.apply() 
    # vfunc = np.vectorize(lambda x : 255 if x >210 else 0)   
    # img_b = vfunc(img_b_clahe).astype('uint8')

    # sky_mask = cv2.bitwise_and(sky_mask,img_b)


    # sky_mask = cv2.imread('/opt/ml/input/final-project-level3-cv-17/SkyReplacement/mask_plus_lapl.png',0)


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
    cv2.imwrite('./mask_edge.png',mask_edge)
    mask_edge_hwc=cv2.merge([mask_edge, mask_edge, mask_edge])

    # guided filtering
    # final =replace.guideFilter(img,transfer,mask_edge_hwc,(3,3),0.00000001)

    final = cv2.cvtColor(transfer, cv2.COLOR_RGB2BGR)
    # final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)

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


