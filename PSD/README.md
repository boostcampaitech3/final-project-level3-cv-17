# Environment
```
Python==3.8.5 or 3.8.13
Pytorch==1.8.1
torchvision==0.9.1
CUDA==10.2
```
```
conda create -n pyiqa python=3.8.5
conda activate pyiqa
conda config --append channels trenta3
conda install cudatoolkit=10.2 -c trenta3
conda install cudatoolkit-dev=10.2 -c trenta3
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
```
```
pip install git+https://github.com/chaofengc/IQA-PyTorch.git
```
# Directory
```
├── data
│   ├── BeDDE
│   │   ├── gt
│   │   ├── hazy
│   │   └── mask
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
│           └── part1
│
└── PSD
    ├── pretrained_model
    │   ├── PSD-GCANET (From PSD)
    │   ├── PSD-FFANET (From PSD)
    │   ├── PSD-MSBDN  (From PSD)
    │   ├── dehazeformer-m.pth (From Dehazeformer)
    │   └── Dehazeformer-Pretrain.pth
    └── finetuned_model
        ├── MSBDN-Finetune.pth
        └── Dehazeformer-Finetune.pth
```

# Weights
* `PSD-GCANET`, `PSD-FFANET`, `PSD-MSBDN`, `dehazeformer-m.pth`는 첨부한 github에서 다운받으시면 됩니다. <br>

    * PSD : https://github.com/zychen-ustc/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors <br>
    * Dehazeformer : https://github.com/IDKiro/DehazeFormer <br>

* `Dehazeformer-Pretrain.pth`, `Dehazeformer-Finetune.pth`, `MSBDN-Finetune.pth`는 [구글 드라이브](https://drive.google.com/drive/folders/1IvmgsbyakQMcHsNMrdT3awe0NkuzTsxV)에서 다운받으실 수 있습니다. <br>

# Run (작성 중)

# Files

* our_datasets.py

    * TrainData_label : Train dataset (gt & hazy) <br>
    * TrainData_unlabel : Train dataset (gt_clahe & hazy) : gt_clahe는 CLAHE.py 결과로 얻을 수 있습니다. <br>
    * ValData_label : Valid dataset (gt & hazy) <br>
    * ETCDataset : Test dataset (hazy) <br>

* make_data.py

    * 모든 데이터로 테스트하는 시간이 오래걸려서 각 폴더별로 5개씩 샘플링하여 새로운 데이터폴더를 만드는 코드입니다.

* wandb_setup_sharing.py

    * 이전 대회와 동일하게 해당 파일을 복사한 후에 wandb_setup.py로 이름을 바꿔주시고, api_key와 project_name 넣어주시면 됩니다.
