# PSD 사용 시 참고사항

### 0. Environment
python==3.8.5 <br>
torch==1.7.1 <br>
torchvision==0.8.2 <br>
cuda==11.0 <br>

pyiqa를 쓰실 경우,
torch==1.8.1 <br>
torchvision==0.9.1 <br>
cuda==10.2 <br>

```
conda create -n pyiqa python=3.8.5
conda activate pyiqa
conda config --append channels trenta3
conda install cudatoolkit=10.2 -c trenta3
conda install cudatoolkit-dev=10.2 -c trenta3
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
```

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
│   │   ├── gt_clahe
│   │   └── hazy
│   ├── NH_HAZE
│   │   ├── gt
│   │   └── hazy
│   ├── O_HAZE
│   │   ├── gt
│   │   └── hazy
│   ├── RESIDE_RTTS
│   │   ├── gt_clahe
│   │   └── hazy
│   ├── RESIDE_SOTS_OUT
│   │   ├── gt
│   │   └── hazy
│   └── RESIDE-OTS (특수기호 주의)
│       ├── gt
│       └── hazy
│           ├── part1
│           ├── part2
│           ├── part3
│           └── part4
│
└── PSD
    └── pretrained_model
        ├── PSD-FFANET
        ├── PSD-GCANET
        └── PSD-MSBDN
``` 

### 2. our_datasets.py
TrainData_label : Train dataset (gt & hazy) <br>
TrainData_unlabel : Train dataset (gt_clahe & hazy) : gt_clahe는 CLAHE.py 결과로 얻을 수 있습니다. <br>
ValData_label : Valid dataset (gt & hazy) <br>
ETCDataset : Test dataset (hazy) <br>

### 3. make_data.py
모든 데이터로 테스트하는 시간이 오래걸려서 각 폴더별로 5개씩 샘플링하여 새로운 데이터폴더를 만드는 코드입니다.

### 4. wandb_setup_sharing.py
이전 대회와 동일하게 해당 파일을 복사한 후에 wandb_setup.py로 이름을 바꿔주시고, api_key와 project_name 넣어주시면 됩니다.

### 5. pyiqa
아래 코드는 공식 github 내용입니다.
requirements.txt에 있는 torch와 torchvision을 지워주셔야 합니다.

```
git clone https://github.com/chaofengc/IQA-PyTorch.git
cd IQA-PyTorch
pip install -r requirements.txt
python setup.py develop
```
