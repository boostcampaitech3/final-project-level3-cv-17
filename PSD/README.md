# PSD 사용 시 참고사항

### 0. Environment
python==3.8.5
torch==1.7.1
torchvision==0.8.2
cuda==11.0

### 1. Directory
data, pretraind_model은 아래처럼 구성해놓았습니다.
data는 gdrive에 zip 파일로 올려두었습니다.
```
├── data
│   ├── BeDDE
│   │   ├── gt
│   │   ├── hazy
│   │   └── mask
│   ├── Crawling
│   │   ├── gt
│   │   └── hazy
│   ├── Hidden
│   │   └── hazy
│   ├── NH_HAZE
│   │   ├── gt
│   │   └── hazy
│   ├── O_HAZE
│   │   ├── gt
│   │   └── hazy
│   ├── RESIDE_RTTS
│   │   └── hazy
│   └── RESIDE_SOTS_OUT
│       ├── gt
│       └── hazy
└── PSD
    └── pretrained_model
        ├── PSD-FFANET
        ├── PSD-GCANET
        └── PSD-MSBDN
``` 

따로 추가한 코드
PSD/pretrained_model
PSD/make_data.py # 모든 데이터로 테스트해보기 시간이 오래걸려서 제가 각 폴더별로 7개씩인가? 데이터를 샘플링해서 새로운 데이터폴더를 만드는 코드입니다.(사용안하셔도 됩니다.)

PSD/datasets/our_datasets.py # PSD/datasets/pretrain_datasets.py에 있던 ETCDatasat을 옮겨놓았습니다.

