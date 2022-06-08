# Sky Replacement
**Change your image's sky to sensitive and wonderful sky!**    

![image](https://user-images.githubusercontent.com/90104418/172551134-44e0c518-3156-494a-b9fc-e0813048ec82.png)

##  Dependencies and Installation
```
Python==3.8
Pytorch==1.7.1
torchvision==0.8.0
CUDA==11.0
```
```
$ pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html  # Check Cuda and Torch version
$ pip install mmsegmentation
$ pip install imutils
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
$ pip install tables 
```


## Directory
```
├── data
│   ├── dehazed_images                                    # Input images 
│   │   ├── dehazed_image_1.jpg
│   │   ├── dehazed_image_1.jpg
│   │   └──  .  .  .
│   ├── sky_images
│   │   ├── database                                      # reference sky images and masks - download dataset ( check below dataset section )
│   │   │   ├── img
│   │   │   │   ├── 0001.jpg
│   │   │   │   ├── 0002.jpg
│   │   │   │   └──  .  .  .
│   │   │   ├── mask
│   │   │   │   ├── 0001.png
│   │   │   │   ├── 0002.png
│   │   │   │   └──  .  .  .
│   ├── sky_replacement                                   # sky replacement final outputs ( incerement jpg path )
│   │   ├── result1.jpg
│   │   ├── result2.jpg
│   │   └──  .  .  .
│   └── sky_replacement_check                             # main.py check option make 6 type images which in sky replacement process
│       ├── 1.png
│       ├── 2.png
│       ├── 3.png
│       ├── 4.png
│       ├── 5.png
│       └── 6.png
│ 
│ 
│
└── SkyReplacement
    ├── main.py
    ├── SkySegmentation
    │   ├── preprocess.py
    │   ├── filter_seg.py
    │   ├── clear_edge.py
    │   ├── mmseg_config
    │   │   ├── configs
    │   │   ├── test.py
    │   │   ├── utils.py
    │   │   ├── train.py
    │   │   ├── wandb
    │   │   ├── work_dirs
    │   ├── checkpoint
    │   │   ├── segformer.py
    │   │   ├── scene_parsing_weight.pth                  # scene parsing model weight
    │   │   └── segmentation_weight.pth                   # sky segmentation model weight
    └── Replacement
         ├── replace.py
         └── SkySelection
               ├── onehotmap.py
               ├── spp.py
               ├── skyselect.py
               ├── sky_db.h5                              # only scene parsing histogram data
               ├── sky_db_clip.h5                         # only clip score data
               └── sky_db_hist_clip.h5                    # scene parsing histogram and clip score data
```
## Weights
- 학습된 Sky Segmentation Weight는 [Gdrive](https://drive.google.com/file/d/1uxf2Llbt0tEoatoIuVsEhqdA2ty6L7Zz/view?usp=sharing)에서 다운 받으실 수 있습니다. 해당 파일은 checkpoint directory에 넣어주시면 됩니다.
- Scene parsing을 위한 모델의 경우[ MMSegmentation 깃허브](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/swin)에서 다운로드 후 convert해서 사용해주시면 됩니다.

## Run
* Sky Replacement Run
```
python main.py --image_path {IMAGE_PATH} --option "a dark night sky"
```
```
optional arguments:
  --image_path IMAGE_PATH
                        path to a dehazed image  ( required ) 
  --sky_path SKY_PATH   path to a sky image ( optional )
  --challenge           If now challenge case, do model
  --check               If check, you can check 6 images which in sky replacement process
  --option OPTION       user select optoin ( ex - 'a pink sky')
```

* Sky Segmentation Model Train
```
cd ./SkySegmentation/mmseg_config
python train.py # 해당 디렉토리 readme참고
```
* Sky Selection DB Generation
```
cd ./Replacement/SkySelection
python skyselect.py
```






## Dataset
[Optimized Sky Dataset- ADE 20k](https://console.cloud.google.com/storage/browser/cvprw2020_sky_seg/public_data/)
[Sky Image Dataset ](https://www.google.com/url?q=http%3A%2F%2Fvllab.ucmerced.edu%2Fytsai%2FSIGGRAPH16%2Fdatabase.zip&sa=D&sntz=1&usg=AOvVaw2zmA3AdJafXUARCFddv1pM) -- put database in data/sky_images/ 
## Reference
**SkyReplacement**

[Adobe Research Sky Replacement Paper](https://sites.google.com/site/yihsuantsai/research/siggraph16-sky)      
[SkyReplacement Repo](https://github.com/HiveYuan/Sky-Replacement)

**Sky Segmentation**

[Sky Segmentation Repo(filter)](https://github.com/OwYeong/SkySegmentationPython)     
[sky optimization](https://github.com/google/sky-optimization)
