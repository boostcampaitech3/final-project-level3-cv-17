# 미세먼지 없는 맑은 사진
### Streamlit & Fastapi 데모 실행
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
    
(영상 )
