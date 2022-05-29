from spp import SpatialPyramidPooling
from onehotmap import label_to_one_hot_label
import numpy as np
import cv2
import mmcv
import torch
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset
from mmcv.runner import load_checkpoint
import glob
import pandas as pd
import os
import torch
from typing import Optional

def build_scene_parsing_model():
    cfg = '/opt/ml/input/final-project-level3-cv-17/SkyReplacement/SkySegmentation/mmseg_config/configs/_base_/upernet_swinL.py'
    checkpoint = '/opt/ml/input/final-project-level3-cv-17/SkyReplacement/SkySegmentation/mmseg_config/configs/_base_/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K_20210526_211650-762e2178.pth'

    cfg = mmcv.Config.fromfile(cfg)
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.cfg = cfg
    model.to('cuda:0')
    model.eval()
    load_checkpoint(model, checkpoint, map_location='cpu')
    dataset = build_dataset(cfg.data.test)
    model.CLASSES = dataset.CLASSES
    return model



def cal_hist(model,img_path):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    parsed_img = inference_segmentor(model, img)
    parsed_img = torch.from_numpy(np.array(parsed_img[0])).type(torch.int64).unsqueeze(0)#.to(device, dtype=torch.float32)
    
    segmap = label_to_one_hot_label(parsed_img, num_classes=150)

    spp = SpatialPyramidPooling(levels=[1,3],mode='avg')

    input_hist = spp(segmap)[0].numpy()

    return input_hist


def sky_hist_db(model,sky_images_paths):
    sky_db = pd.DataFrame()
    sky_hists = []
    sky_paths = []
    mask_paths = []
    for sky_path in glob.glob(sky_images_paths): 
        # opencv imread로 읽을 수 있는 파일 형식만 
        mask_path = sky_path.replace('img','mask')
        try:
            sky_hist = cal_hist(model,sky_path)
        except:
            os.remove(sky_path)
            os.remove(mask_path)
            pass

        sky_paths.append(sky_path)
        sky_hists.append(sky_hist)
        mask_paths.append(mask_path)

    sky_db['img_path'] = sky_paths
    sky_db['hist'] = sky_hists
    sky_db['mask_path'] = mask_paths

    return sky_db

def sky_db_gen(sky_images_paths):
    model = build_scene_parsing_model()
    sky_db = sky_hist_db(model,sky_images_paths)
    # save
    sky_db.to_hdf('sky_db.h5',key='df')# csv, pickle 사용시 손실 발생 1500-> 90

def find_min_sky_rect(sky_mask):
    for index,i in enumerate(sky_mask):
        if sky_mask.shape[1] > len(np.where(i!=0)[0]):
            y_max = index
            break

    x_max = sky_mask.shape[1]    
    return (y_max,x_max,0,0)

def find_max_sky_rect(mask):

	index=np.where(mask!=0)
	index= np.array(index,dtype=int)
	y=index[0,:]
	x=index[1,:]
	c2=np.min(x)
	c1=np.max(x)
	r2=np.min(y)
	r1=np.max(y)
	
	return (r1,c1,r2,c2)

def cal_P(sky_mask):
    height, width = sky_mask.shape[:2]
    P_s = width * height
    P_a = width/height
    return P_a,P_s

def cal_Q(P_I,P_R):
    Q = min(P_I,P_R)/max(P_I,P_R)
    return Q

def determine(ref_sky_mask, input_sky_mask):

    ref_P_a, ref_P_s = cal_P(ref_sky_mask)
    input_P_a, input_P_s = cal_P(input_sky_mask)

    Q_a = cal_Q(input_P_a,ref_P_a)
    Q_s = cal_Q(input_P_s,ref_P_s)

    if Q_a > 0.5 and Q_s >0.5:
        return True
    else:
        return False


def cal_sim(input_hist, sky_db):
    """_summary_

    Args:
        input (): input hist
        sky_db (dict): sky_db dataframe
    """
    sky_cal = sky_db.copy()
    methods = {'CORREL' :cv2.HISTCMP_CORREL, 'CHISQR':cv2.HISTCMP_CHISQR, 
           'INTERSECT':cv2.HISTCMP_INTERSECT,
           'BHATTACHARYYA':cv2.HISTCMP_BHATTACHARYYA}

    for name, flag in methods.items():
        tmp =[]
        for i, hist in enumerate(sky_cal['hist']):
            hist = hist.astype('float32') # delete
            ret = cv2.compareHist(input_hist, hist, flag)

            if flag == cv2.HISTCMP_INTERSECT: 
                ret = ret/np.sum(input_hist)        

            tmp.append(ret)
        
        sky_cal[f'{name}'] = tmp
    return sky_cal

def select_sky(input_img_path, sky_images_paths, input_sky_mask):
    # model build
    model = build_scene_parsing_model()
    
    # cal hist
    input_hist = cal_hist(model,input_img_path)

    # read sky_db which contain all sky image's hists
    sky_db = pd.read_hdf('/opt/ml/input/final-project-level3-cv-17/SkyReplacement/Replacement/SkySelection/sky_db.h5')

    # calculate similarity between input hist and all sky hists
    sky_cal = cal_sim(input_hist, sky_db)

    # sort by specific sim metric ( default : correaltion )
    sky_cal.sort_values('CORREL',ascending=False)

    # 근데 여기에 추가기준이 필요함. 계산하는
    # 여기서 상위부터 계산하자. 계산해서 되면 ok아니면 넘어가기.
    # 정렬한다음에 위에서부터 계산해서 ok면 가져오기 아니면 다음거로 넘어가기

    # test
    # input_sky_mask = cv2.imread('/opt/ml/input/SkySegmentationPython/555_sample_output.jpg',0)

    for img_path,mask_path in zip(sky_cal['img_path'],sky_cal['mask_path']):
        ref_sky_mask = cv2.imread(mask_path,0)
        if determine(ref_sky_mask, input_sky_mask):
            # select_path = img_path.copy()
            # 함수 밖의 변수를 참조만 하는게 아닌 
            # 할당을 하게 될 경우 unboundlocal error가 발생
            break
    return img_path




if __name__ =="__main__":
    sky_db_gen('/opt/ml/input/final-project-level3-cv-17/data/sky_images/database/img/*')
    # k = select_sky('/opt/ml/input/final-project-level3-cv-17/data/dehazed_images/555.png','/opt/ml/input/final-project-level3-cv-17/data/sky_images/database/img/*','/opt/ml/input/SkySegmentationPython/555_sample_output.jpg')
    # print(k)
