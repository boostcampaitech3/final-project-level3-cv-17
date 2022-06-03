from .spp import SpatialPyramidPooling
from .onehotmap import label_to_one_hot_label
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
from typing import Optional
import clip
from PIL import Image
from tqdm import tqdm

#####################################################################################################
##################################### Scene Parsing Histogram #######################################
#####################################################################################################

def cal_hist(model,img_path):

    if type(img_path) == 'str':
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    else: # 이미지 자체가 들어왔을 때
        img = img_path 

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


def cal_sim(input_hist, sky_db):
    """cal histogram similarity

    Args:
        input_hist (list): input hist ( length : 1500 )
        sky_db (DataFrame): sky_db dataframe
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
        
        sky_cal[f'{name}'] = map(lambda x : round(x,2),tmp)
    return sky_cal


#####################################################################################################
############################################# Filtering #############################################
#####################################################################################################
### Mask Processing ( Intersection ) ###
def mask_intersection(input_mask, ref_mask):
    height, width = input_mask.shape
    re_ref_mask = cv2.resize(ref_mask,(width,height))
    bwa = cv2.bitwise_and(re_ref_mask,input_mask)
    mask_bool = (bwa == input_mask)

    ####################################### can change treshold value ( the number of intersection pixels)#######################
    if len(np.where(mask_bool==False)[0]) < 100:
        return True
    else:
        return False

### Mask Processing ( Rectengular ) ###
def find_min_sky_rect(sky_mask):
    for index,i in enumerate(sky_mask):
        if sky_mask.shape[1] > len(np.where(i!=0)[0]):
            break
    y_max = index
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

# sky mask가 너무 작다면 -rect방식
def detr_size(ref_sky_mask):
    y_max, _, _, _ = find_min_sky_rect(ref_sky_mask)
    height = ref_sky_mask.shape[0]
    print(y_max / height)
    if y_max / height < 0.2:
        return False
    else:
        return True

### Resolution & Aspect Ratio Filtering ###
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

    # print(Q_a,Q_s)

    if Q_a > 0.5 and Q_s > 0.25:
        return True
    else:
        return False

#####################################################################################################
############################################# Selection #############################################
#####################################################################################################

def select_sky(input_img_path, input_sky_mask, option, num=5 ,layout = False):
    """Sky Selection Algorithm 
    1. Sorted by option Clip score
    2. filtering mask intersection, resolution, aspect ratio
    3. 5 images were recommended

    Args:
        input_img_path (str): input image path
        input_sky_mask (np ndarray): input image sky segmentation mask
        option (str): option of user selection
        num (int,optional) : The number of recommended sky images. Defaults to 5.
        layout (bool, optional): If layout=True, can use layout descriptor. Defaults to False.

    Returns:
        Recommend_list(list): list of all recomendation image paths. If retrun list's lenght < num , there are not enough recommended images.
    """
    sky_cal = pd.read_hdf('app/weights/sky_db_clip.h5')
    
    # sort by user option
    sky_cal = sky_cal.sort_values(by=option, ascending=False)

    # filtering & num images recommendation
    recommend_list = []
    for img_path,mask_path in zip(sky_cal['img_path'],sky_cal['mask_path']):
        ref_sky_mask = cv2.imread(mask_path,0)
        if determine(ref_sky_mask, input_sky_mask) and mask_intersection(input_sky_mask, ref_sky_mask):
            recommend_list.append(img_path)
        if len(recommend_list) == num:
            break          
    return recommend_list 

# def select_sky(input_img_path, input_sky_mask):
#     # model build
#     model = build_scene_parsing_model()
    
#     # cal hist
#     input_hist = cal_hist(model,input_img_path)

#     # read sky_db which contain all sky image's hists
#     sky_db = pd.read_hdf('/opt/ml/input/final-project-level3-cv-17/SkyReplacement/Replacement/SkySelection/sky_db.h5',model ,preprocess)

#     # calculate similarity between input hist and all sky hists
#     sky_cal = cal_sim(input_hist, sky_db)

#     # sort by specific sim metric ( default : correaltion )
#     sky_cal.sort_values('CORREL',ascending=False)
    
#     # test
#     # input_sky_mask = cv2.imread('/opt/ml/input/SkySegmentationPython/555_sample_output.jpg',0)

#     for img_path,mask_path in zip(sky_cal['img_path'],sky_cal['mask_path']):
#         ref_sky_mask = cv2.imread(mask_path,0)
#         print(determine(ref_sky_mask, input_sky_mask) )
#         if determine(ref_sky_mask, input_sky_mask):
#             print('**************')
#             print(detr_size(ref_sky_mask))
#             # select_path = img_path.copy()
#             # 함수 밖의 변수를 참조만 하는게 아닌 
#             # 할당을 하게 될 경우 unboundlocal error가 발생
#             break


#     return img_path




if __name__ =="__main__":
    clip_score('/opt/ml/input/final-project-level3-cv-17/data/sky_images/database/img/*')
    # sky_db_gen('/opt/ml/input/final-project-level3-cv-17/data/sky_images/database/img/*')
    # k = select_sky('/opt/ml/input/final-project-level3-cv-17/data/dehazed_images/555.png','/opt/ml/input/final-project-level3-cv-17/data/sky_images/database/img/*','/opt/ml/input/SkySegmentationPython/555_sample_output.jpg')
    # print(k)