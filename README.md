# 미세먼지 없는 맑은 사진
### Streamlit & Fastapi 데모 실행
1. 모델 Weights 다운로드
    - `serving/app` 안에 weights 폴더를 만듭니다. 구조는 다음과 같습니다.
    
      ```bash
      └── weights
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
