
## 미세먼지 없는 맑은 사진
* CV 17조 MG세대
* Presentation Slide : https://drive.google.com/drive/folders/1_bYN3mC4viJHQI5G_y1e7f1iK3_IRDwW?usp=sharing

## Project Abstract
* Problem Definition
    * 특별한 날, 특별한 장소에서 미세먼지 때문에 원하는 사진을 찍지 못하거나 안 찍는 상황이 생김
    * 하지만 보정에 대한 전문 지식이 부족하거나 필터가 제한되는 경우 사용자가 원하는 방향으로 사진을 보정하기 어려웠음

* Main features
    * 사용자가 업로드한 Hazy한 이미지를 미세먼지가 없는 선명한 사진으로 변환
    * 사용자는 원하는 Keyword의 하늘 이미지를 업로드한 사진에 합성

## Member Introduction
|팀원|Github|역할|
| :--------: | :--------: | :--------: |
|[T3078] 민선아|[@seonahmin](https://github.com/seonahmin)|Image Dehazing|
|[T3101] 백경륜|[@baekkr95](https://github.com/baekkr95)|Product Serving|
|[T3139] 이도연|[@omocomo](https://github.com/omocomo)|Product Serving|
|[T3177] 이효석|[@hyoseok1223](https://github.com/hyoseok1223)|PM, Sky Replacement|
|[T3179] 임동우|[@Dongwoo-Im](https://github.com/Dongwoo-Im)|Image Dehazing|

## Service Architecture
![image](https://user-images.githubusercontent.com/81875412/172397327-77f34979-b0b4-45f7-992f-b0e126c6d10b.png)

## Demo
* Dependencies and Installation

1. 모델 Weights 다운로드
    - `serving/app` 안에 weights 폴더를 만듭니다. 구조는 다음과 같습니다.
    
      ```bash
      weights
      ├── Dehazing
      │   └── Dehazeformer-Finetune.pth
      └── Sky
          ├── SkyDB
          │   └── sky_db_clip.h5
          └── SkySegmentation
              └── seg_epoch_20.pth
      ``` 
    - weights는 [구글 드라이브](https://drive.google.com/drive/folders/1cGudVyyesPung0HcA_IXPMSXmHceMCX-?usp=sharing)에서 다운로드 받을 수 있습니다.

2. 실행 시키기
    ```
    cd serving
    make -j 2 run_app
    ```
    - Makefile run_client의 streamlit이 실행되면 해당 url에서 동작을 확인할 수 있습니다.


3. 데모 영상
    - 이미지 업로드 후 Dehazing 실행

![시연-첫부분](https://user-images.githubusercontent.com/48708496/172543350-dedb428c-ce56-4517-851b-8ae2da2a11d0.gif)

    - 원하는 하늘 사진 선택 후 합성
    
(영상)

## Model Process
![image](https://user-images.githubusercontent.com/90104418/172589792-e65c3092-38ea-42cc-8fdc-de7f4b548db1.png)

## Reference
* Image Dehazing
    * PSD : https://github.com/zychen-ustc/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors
    * DehazeFormer : https://github.com/IDKiro/DehazeFormer
    * AECR-Net : https://github.com/GlassyWu/AECR-Net
    * IQA-Pytorch : https://github.com/chaofengc/IQA-PyTorch

* Sky Replacement
    * SegFormer : https://github.com/NVlabs/SegFormer
    * Sky Is Not the Limit Paper : https://sites.google.com/site/yihsuantsai/research/siggraph16-sky
    * Sky Segmentation Reference Repo : https://github.com/OwYeong/SkySegmentationPython
    * Sky optimization : https://github.com/google/sky-optimization

* DataSet
    * RESIDE : https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=0
    * MRFID : http://www.vistalab.ac.cn/MRFID-for-defogging/
    * BeDDE : https://github.com/xiaofeng94/BeDDE-for-defogging
    * Optimized Sky Dataset- ADE 20k : https://console.cloud.google.com/storage/browser/cvprw2020_sky_seg/public_data/
    * Sky Image Dataset : https://www.google.com/url?q=http%3A%2F%2Fvllab.ucmerced.edu%2Fytsai%2FSIGGRAPH16%2Fdatabase.zip&sa=D&sntz=1&usg=AOvVaw2zmA3AdJafXUARCFddv1pM

