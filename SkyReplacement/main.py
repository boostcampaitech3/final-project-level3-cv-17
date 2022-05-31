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

# 오류가 나는 이유를 생각해보면 결국 코드를 가져오는 것이기 때문이다.
# 여기에 spatial 모듈을 이 디렉토리에서 찾으면 일단 없고
# 따로 PYTHON PTAH에 넣어놓은 것도 아니고
# 설치 의존적 패키지도 아니니까 오류나는게 어떻게 보면 당연하다
# 따라서 파이썬 프로젝트 코드 작성시 보통 어떻게 하는지 궁그맿서 찾아봤는데
# 2번같이 해서 혹은 setup을 해서 어디서든 갖올 수 있게 해두면 된다.
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
    
    args =  parser.parse_args()

    img=cv2.cvtColor(cv2.imread(args.image_path,1), cv2.COLOR_BGR2RGB)
    # sky=cv2.cvtColor(cv2.imread(args.sky_path,1), cv2.COLOR_BGR2RGB)

    start_time = time.time()        

    print('Start')

    if args.challenge :
        config_file = '/opt/ml/input/final-project-level3-cv-17/SkyReplacement/SkySegmentation/checkpoint/segformer.py'
        checkpoint_file = '/opt/ml/input/final-project-level3-cv-17/SkyReplacement/SkySegmentation/checkpoint/epoch_34.pth'
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

    # filter based
    else:
        sky_mask = process_image_or_folder(args)
        print(sky_mask.shape)
        print(type(sky_mask))
    
    sz=img.shape

    # 직접 마스크 지정
    # import pickle
    # with open('/opt/ml/input/final-project-level3-cv-17/SkyReplacement/SkySegmentation/mmseg_config/mask.pkl', 'rb') as f:
    #     sky_mask = pickle.load(f)
    # print(sky_mask.shape)
    # print(type(sky_mask))
    # sky_mask = sky_mask.astype(np.uint8) 

    # select the sky
    if args.sky_path:
        sky_path = args.sky_path
    else:
        sky_path = select.select_sky(args.image_path, SKY_IMAGES_PATH, sky_mask)

    print(sky_path)
    sky=cv2.cvtColor(cv2.imread(sky_path,1), cv2.COLOR_BGR2RGB)
    mask_path = sky_path.replace('img','mask')
    ref_mask = cv2.imread(mask_path,0)
    # 여기 어딘가에서 resize가 필요함

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


