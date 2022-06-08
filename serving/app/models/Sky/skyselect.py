import numpy as np
import cv2
import pandas as pd
from typing import Optional
from PIL import Image

#####################################################################################################
############################################# Filtering #############################################
#####################################################################################################

### Mask Processing ( Intersection ) ###
def mask_intersection(input_mask, ref_mask):
    height, width = input_mask.shape
    re_ref_mask = cv2.resize(ref_mask,(width,height))
    bwa = cv2.bitwise_and(re_ref_mask,input_mask)
    mask_bool = (bwa == input_mask)

    ### can change treshold value ( the number of intersection pixels) ###
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
    # print(y_max / height)
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

    if Q_a > 0.45 and Q_s > 0.25:
        return True
    else:
        return False

#####################################################################################################
############################################# Selection #############################################
#####################################################################################################

def select_sky(input_img_path, input_sky_mask, option, num=5):
    """Sky Selection Algorithm 
    1. Sorted by option Clip score
    2. filtering mask intersection, resolution, aspect ratio
    3. 5 images were recommended
    Args:
        input_img_path (str): input image path
        input_sky_mask (np ndarray): input image sky segmentation mask
        option (str): option of user selection
        num (int,optional) : The number of recommended sky images. Defaults to 5.
    Returns:
        Recommend_list(list): list of all recomendation image paths. If retrun list's lenght < num , there are not enough recommended images.
    """
    sky_cal = pd.read_hdf('app/weights/Sky/SkyDB/sky_db_clip.h5')
    
    # sort by user option
    # sky_cal = sky_cal.sort_values(by = option,ascending=False)

    # filter by user option and increase diversity
    sky_cal = sky_cal[sky_cal[option]>0.35]
    sky_cal = sky_cal.sample(frac=1).reset_index(drop=True)

    # filtering & num images recommendation
    recommend_list = []
    for img_path,mask_path in zip(sky_cal['img_path'],sky_cal['mask_path']):
        ref_sky_mask = cv2.imread(mask_path,0)
        if determine(ref_sky_mask, input_sky_mask) and mask_intersection(input_sky_mask, ref_sky_mask):
            recommend_list.append(img_path)
        if len(recommend_list) == num:
            break           

    # print(recommend_list)
    return recommend_list 