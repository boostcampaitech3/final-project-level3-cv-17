
## ๐ค๏ธ๋ฏธ์ธ๋จผ์ง ์๋ ๋ง์ ์ฌ์ง๐ท 
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/> <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white"/> <img src="https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white"/>                                                                                                                                                                                                                                              
##### ๐ Presentation Slide : https://drive.google.com/drive/folders/1_bYN3mC4viJHQI5G_y1e7f1iK3_IRDwW?usp=sharing
## ๐TEAM
### CV 17์กฐ MG์ธ๋
|๋ฏผ์ ์|๋ฐฑ๊ฒฝ๋ฅ|์ด๋์ฐ|์๋์ฐ|์ดํจ์|
| :--------: | :--------: | :--------: | :--------: | :--------: |
|<img src="https://user-images.githubusercontent.com/78402615/172766340-8439701d-e9ac-4c33-a587-9b8895c7ed07.png" width="120" height="120"/>|<img src="https://user-images.githubusercontent.com/78402615/172766371-7d6c6fa3-a7cd-4c21-92f2-12d7726cc6fc.png" width="120" height="120"/>|<img src="https://user-images.githubusercontent.com/78402615/172784450-628b30a3-567f-489a-b3da-26a7837167af.png" width="120" height="120"/>|<img src="https://user-images.githubusercontent.com/78402615/172766404-7de4a05a-d193-496f-9b6b-5e5bdd916193.png" width="120" height="120"/>|<img src="https://user-images.githubusercontent.com/78402615/172766321-3b3a4dd4-7428-4c9f-9f91-f69a14c9f8cc.png" width="120" height="120"/>|
|[@seonahmin](https://github.com/seonahmin)|[@baekkr95](https://github.com/baekkr95)|[@omocomo](https://github.com/omocomo)|[@Dongwoo-Im](https://github.com/Dongwoo-Im)|[@hyoseok1223](https://github.com/hyoseok1223)|
|Image Dehazing|Product Serving|Product Serving|Image Dehazing|PM, Sky Replacement|

## ๐ Project Abstract
* Problem Definition
    * ํน๋ณํ ๋ , ํน๋ณํ ์ฅ์์์ ๋ฏธ์ธ๋จผ์ง ๋๋ฌธ์ ์ํ๋ ์ฌ์ง์ ์ฐ์ง ๋ชปํ๊ฑฐ๋ ์ ์ฐ๋ ์ํฉ์ด ์๊น
    * ํ์ง๋ง ๋ณด์ ์ ๋ํ ์ ๋ฌธ ์ง์์ด ๋ถ์กฑํ๊ฑฐ๋ ํํฐ๊ฐ ์ ํ๋๋ ๊ฒฝ์ฐ ์ฌ์ฉ์๊ฐ ์ํ๋ ๋ฐฉํฅ์ผ๋ก ์ฌ์ง์ ๋ณด์ ํ๊ธฐ ์ด๋ ค์ ์

* Main features
    * ์ฌ์ฉ์๊ฐ ์๋ก๋ํ Hazyํ ์ด๋ฏธ์ง๋ฅผ ๋ฏธ์ธ๋จผ์ง๊ฐ ์๋ ์ ๋ชํ ์ฌ์ง์ผ๋ก ๋ณํ
    * ์ฌ์ฉ์๋ ์ํ๋ Keyword์ ํ๋ ์ด๋ฏธ์ง๋ฅผ ์๋ก๋ํ ์ฌ์ง์ ํฉ์ฑ

![image](https://user-images.githubusercontent.com/48708496/173018846-fab41312-88c9-4e31-bf4d-7161060cd7c7.png)



<!-- ## Member Introduction

|ํ์|Github|์ญํ |
| :--------: | :--------: | :--------: |
|[T3078] ๋ฏผ์ ์|[@seonahmin](https://github.com/seonahmin)|Image Dehazing|
|[T3101] ๋ฐฑ๊ฒฝ๋ฅ|[@baekkr95](https://github.com/baekkr95)|Product Serving|
|[T3139] ์ด๋์ฐ|[@omocomo](https://github.com/omocomo)|Product Serving|
|[T3177] ์ดํจ์|[@hyoseok1223](https://github.com/hyoseok1223)|PM, Sky Replacement|
|[T3179] ์๋์ฐ|[@Dongwoo-Im](https://github.com/Dongwoo-Im)|Image Dehazing| -->

## ๐ฅ Service Architecture
![๊นํ์๋น์ค์ฌ์ง](https://user-images.githubusercontent.com/48708496/172779913-8815fccf-321d-4ba3-a7b2-8d1d13ef2549.jpg)

## ๐ Demo
0. Dependencies and Installation
   - `environment.yaml`์ name, prefix ์ค์ 
   - ๊ฐ์ํ๊ฒฝ ์์ฑ ๋ฐ install
   
      ```
      cd serving
      conda env create -f environment.yaml # ๊ฐ์ํ๊ฒฝ ์์ฑ + install
      conda activate serving # environment.yaml์ name
      pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
      ```
      
1. ๋ชจ๋ธ Weights ๋ค์ด๋ก๋
    - `serving/app` ์์ weights ํด๋๋ฅผ ๋ง๋ญ๋๋ค. ๊ตฌ์กฐ๋ ๋ค์๊ณผ ๊ฐ์ต๋๋ค.
    
      ```bash
      weights
      โโโ Dehazing
      โ   โโโ Dehazeformer-Finetune.pth
      โโโ Sky
          โโโ SkyDB
          โ   โโโ sky_db_clip.h5
          โโโ SkySegmentation
              โโโ seg_epoch_20.pth
      ``` 
    - weights๋ [๊ตฌ๊ธ ๋๋ผ์ด๋ธ](https://drive.google.com/drive/folders/1cGudVyyesPung0HcA_IXPMSXmHceMCX-?usp=sharing)์์ ๋ค์ด๋ก๋ ๋ฐ์ ์ ์์ต๋๋ค.

2. DB URL ์ค์ 
   - `serving/app/db/__init__.py`์ MONGO_URL ์ค์ 

3. ์คํ ์ํค๊ธฐ
    ```
    cd serving
    make -j 2 run_app
    ```
    - Makefile run_client์ streamlit์ด ์คํ๋๋ฉด ํด๋น url์์ ๋์์ ํ์ธํ  ์ ์์ต๋๋ค.


3. ๋ฐ๋ชจ ์์
<h4 align="center">์ด๋ฏธ์ง ์๋ก๋ ํ Dehazing ์คํ</h4>

![๊นํ๋ธ๋ฐ๋ชจ์๋ถ๋ถ](https://user-images.githubusercontent.com/48708496/172776811-ad304a19-2bcd-40b6-ad65-721c10ff2875.gif)

<h4 align="center">์ํ๋ ํ๋ ์ฌ์ง ์ ํ</h4>

![๊นํ๋ธ์ค๊ฐ๋ฐ๋ชจ](https://user-images.githubusercontent.com/48708496/172778234-978d739f-09cf-400a-820c-44ba229d140f.gif)


<h4 align="center">ํ๋ ํฉ์ฑ๊น์ง ์๋ฃ๋ ์ต์ข ์ด๋ฏธ์ง</h4>

![๊นํ๋ธ๋ง์ง๋ง๋ฐ๋ชจ](https://user-images.githubusercontent.com/48708496/172778813-a33ceff5-ce4d-4289-978f-a066520b4492.gif)


## ๐ผ Model Process
![image](https://user-images.githubusercontent.com/90104418/172589792-e65c3092-38ea-42cc-8fdc-de7f4b548db1.png)

## ๐ Reference
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

