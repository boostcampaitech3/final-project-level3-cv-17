## 미세먼지 없는 맑은 사진
* Naver Boostcamp AI Tech 3rd final project
* CV 17조 MG세대
* Demo Video : 
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

## Streamlit & Fastapi Demo
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

## Model Process
![image](https://user-images.githubusercontent.com/81875412/172397492-34a7450e-32e4-4f45-a9a2-87b4a43a07f2.png)

## Image Dehazing
* Dependencies and Installation
```
Python==3.8.5 or 3.8.13
Pytorch==1.8.1
torchvision==0.9.1
CUDA==10.2
```
* Pretrain (작성 중)
```
cd PSD
python main.py
```
* Finetune (작성 중)
```
cd PSD
python finetune.py --I_I2_loss --unlabel_index_dir=''
```
## Sky Replacement
* Dependencies and Installation

* Train

## Reference
* Image Dehazing
    * PSD : https://github.com/zychen-ustc/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors
    * DehazeFormer : https://github.com/IDKiro/DehazeFormer
    * AECR-Net : https://github.com/GlassyWu/AECR-Net
    * IQA-Pytorch : https://github.com/chaofengc/IQA-PyTorch

* Sky Replacement
    * SegFormer : https://github.com/NVlabs/SegFormer
