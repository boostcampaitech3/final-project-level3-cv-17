import cv2
import torch
from spatial import SpatialPyramidPooling
from onehotmap import label_to_one_hot_label
import pickle
import numpy as np
import mmcv
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset
from mmcv.runner import load_checkpoint
import glob
import pandas as pd
import os


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
    # result = inference_segmentor(model, img)
    return model



# input으로 들어오는 이미지 하나에 대해서 SPP 계산
# sky_images에 대해서는 미리 계산되어 있어야함.

# 저 둘간의 유사도를 구해야함.
# 그리고 그걸로 소팅한다음에 상위 몇개 보여주면 됨.

# '/opt/ml/input/final-project-level3-cv-17/data/skyseg/img_dir/test/ADE_val_00000002.jpg'

def cal_hist(model,img_path):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    parsed_img = inference_segmentor(model, img)
    parsed_img = torch.from_numpy(np.array(parsed_img[0])).type(torch.int64).unsqueeze(0)#.to(device, dtype=torch.float32)
    
    segmap = label_to_one_hot_label(parsed_img, num_classes=150)

    spp = SpatialPyramidPooling(levels=[1,3],mode='avg')

    input_hist = spp(segmap)[0].numpy()

    return input_hist


# def input_hist(input_path):
    
#     ###### input ########
#     input_img = cv2.imread(input_path)
#     input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
#     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     input_segmap = mmseg_inference(cfg,ckpt,input_img)

#     input_segmap = torch.from_numpy(np.array(input_segmap[0])).type(torch.int64).unsqueeze(0)#.to(device, dtype=torch.float32)
#     input_segmap = label_to_one_hot_label(input_segmap, num_classes=150)

#     spp = SpatialPyramidPooling(levels=[1,3],mode='avg')

#     input_hist = spp(input_segmap)[0].numpy()

#     return input_hist

## 얘는 한 번만 해두고 이 데이터 프레임 불러오는 방식으로 코드 수정필요
def sky_hist_db(model,sky_images_paths):
    sky_db = pd.DataFrame()
    sky_hists = []
    sky_paths = []
    mask_paths = []
    for sky_path in glob.glob(sky_images_paths): # '/opt/ml/input/final-project-level3-cv-17/data/sky_images/*'
        # opencv imread로 읽을 수 있는 파일 형식만 
        mask_path = sky_path.replace('img','mask')
        try:
            sky_hist = cal_hist(model,sky_path)
        except:
            # os.remove(sky_path)
            # os.remove(mask_path)
            pass
        # sky_img = cv2.imread(sky_path)
        # sky_img = cv2.cvtColor(sky_img,cv2.COLOR_BGR2RGB)
        # sky_segmap = mmseg_inference(cfg,ckpt,sky_img)

        # sky_segmap = torch.from_numpy(np.array(sky_segmap[0])).type(torch.int64).unsqueeze(0)#.to(device, dtype=torch.float32)
        # sky_segmap = label_to_one_hot_label(sky_segmap, num_classes=150)
        # spp = SpatialPyramidPooling(levels=[1,3],mode='avg')
        # sky_hist = spp(sky_segmap)

        sky_paths.append(sky_path)
        sky_hists.append(sky_hist)
        mask_paths.append(mask_path)

    sky_db['img_path'] = sky_paths
    sky_db['hist'] = sky_hists
    sky_db['mask_path'] = mask_paths

    return sky_db

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

def sky_db_gen(sky_images_paths):
    model = build_scene_parsing_model()
    sky_db = sky_hist_db(model,sky_images_paths)
    # save
    sky_db.to_hdf('data.h5',key='df')# csv, pickle 사용시 손실 발생 1500-> 90


# 기본적으로 몇개를 추천해주고, 맘에 안든다고 하면 고른 걸로 바꿔줄 수 있게 해주면 되겠다.
def select_sky(input_img_path, sky_images_paths, input_sky_mask):
    model = build_scene_parsing_model()
    
    input_hist = cal_hist(model,input_img_path)
    # sky_db = sky_hist_db(model,sky_images_paths)
    # # save
    # sky_db.to_csv('sky_db.csv')
    sky_db = pd.read_hdf('data.h5')
    sky_cal = cal_sim(input_hist, sky_db)
    # 이걸 뭔가 기준으로 정렬하면 됨. 
    sky_cal.sort_values('CORREL',ascending=False)#.loc[0]['img_path']
    # 근데 여기에 추가기준이 필요함. 계산하는
    # 여기서 상위부터 계산하자. 계산해서 되면 ok아니면 넘어가기.
    # 정렬한다음에 위에서부터 계산해서 ok면 가져오기 아니면 다음거로 넘어가기

    # test
    input_sky_mask = cv2.imread('/opt/ml/input/SkySegmentationPython/555_sample_output.jpg',0)

    for img_path,mask_path in zip(sky_cal['img_path'],sky_cal['mask_path']):
        ref_sky_mask = cv2.imread(mask_path,0)
        if determine(ref_sky_mask, input_sky_mask):
            sky_path = img_path
            break
    return sky_path
    # 애초에 sky_db에 mask도 미리 만들어서 경로를 넣어둬야 겠다. 그러면 가져올 때 inference할 필요가 없자나




# input hist랑 sky hists 목록 받아서
# 1대 다 계산 하고 정렬 후 추천
def cal_sim(input_hist, sky_db):
    """_summary_

    Args:
        input (_type_): input image & hist
        sky_db (dict): db csv 
    """
    sky_cal = sky_db.copy()
    methods = {'CORREL' :cv2.HISTCMP_CORREL, 'CHISQR':cv2.HISTCMP_CHISQR, 
           'INTERSECT':cv2.HISTCMP_INTERSECT,
           'BHATTACHARYYA':cv2.HISTCMP_BHATTACHARYYA}

    for name, flag in methods.items():
        tmp =[]
        for i, hist in enumerate(sky_cal['hist']):
            #---④ 각 메서드에 따라 img1과 각 이미지의 히스토그램 비교
            # 나중에 지우기
            hist = hist.astype('float32')
            ret = cv2.compareHist(input_hist, hist, flag)

            

            # cv2.error: OpenCV(4.5.5) :-1: error: (-5:Bad argument) in function 'compareHist'
            # > Overload resolution failed:
            # >  - H2 data type = 23 is not supported
            # >  - Expected Ptr<cv::UMat> for argument 'H2'
            # long long 의경우 23 즉 float 32간 연산을해야하는데 지금 하나가 16d이라서 난 오류
            if flag == cv2.HISTCMP_INTERSECT: #교차 분석인 경우 
                ret = ret/np.sum(input_hist)        #비교대상으로 나누어 1로 정규화

            tmp.append(ret)
        
        sky_cal[f'{name}'] = tmp
    return sky_cal





if __name__ =="__main__":
    # model = build_scene_parsing_model()
    # img_path = '/opt/ml/input/final-project-level3-cv-17/data/sky_images/database/img/0001.png'
    # print(len(cal_hist(model,img_path)))
    # sky_db_gen('/opt/ml/input/final-project-level3-cv-17/data/sky_images/database/img/*')
    k = select_sky('/opt/ml/input/final-project-level3-cv-17/data/dehazed_images/555.png','/opt/ml/input/final-project-level3-cv-17/data/sky_images/database/img/*','/opt/ml/input/SkySegmentationPython/555_sample_output.jpg')
    print(k)




# 한 픽셀에 대해서 n개의 카테고리에 대한 모든 값들이 모여있는 response map 즉, 그냥 리턴되는  response map
# 이거에 대해서 acg pool을 통해서 뽑고
# img하나에 대해서 spp를 통해서 할건데 여기서 9개의 그리드로 나눠서 각각에 히스토그램 뽑ㅂ고, 전체 히스토그램 뽑고 얘내 concat하기


# 여러개의 이미지에 대해서 이게 동작해야하는 건가..? 그건 아닌 것 같음. 


# 이건 모든 reference image와 우리의 input image에 대해서 동작하면 되고, 그 결과를 



# 잠만, 다시 정리 이미지를 9개 그리드로 나누기 + 이미지 전체 이렇게 총 10개의 patch에 대해서
# H라는 히스토그램을 뽑을 거다. 이 과정이 H라는게 우리가 위에서 봤던 저 avg pool로 얻는게 아닐까 싶다.

# 1. SPP를 통해서 뽑는다. 하나의 전체 이미지에 대해서도 0인 픽셀 1인 픽셀 2인 픽셀 등등 있을 텐데, 개수가 제일 많은게 나오는거 겠지?
# 2. 근데 생각해보니 amx pooling이 아니라 avg pooling이라서 그냥 계산이긴함
# 3. 어쨋든 클래스 개수만큼 값이 하나 나오고 각 패치에 대해서도 클래스 개수만큼 값이 나온다.
# 4. 클래스의 개수만큼 나오게 되면 그게 하나의 H가 되고, sky 클래스에 대한 확률 값 하나, ground 클래스에 대한 확률값 하나
#  이렇게 해서 총 11개의 클래스 개수만큼 값들이 나온다. 하나의 패치에 대해서 그것들을 전부 concat 그럼 11x10이 되겠네

# 이걸 다른 이미지에 대해서도 계산하고 이 둘을 히스토그램을 계산하는거.


