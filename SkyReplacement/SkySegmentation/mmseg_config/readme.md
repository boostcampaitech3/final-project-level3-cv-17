# MMSegmentation
## configs file using directions (Korean)

- configs 디렉토리 안의 model, dataset, schedule 각각의 폴더 안에 사용하고자 하는 config 설정을 한다.
- base 디렉토리 안의 파일들에는 model, dataset, schedule, runtime에서 설정한 config를 가져온다.해당 .py 파일들은 train, test 과정에서 load한다.
- 모든 config 세팅은 앞서 말한 것 처럼 _base_파일에서 병합된다. 이 때, 추가적으로 config를 설정하고 싶거나 수정하고 싶다면 _base_에서 overwrite를 해주면 된다.

## Convert Pretrained Weights to MMSeg Weights
일부 모델들의 경우 사전학습된 weights를 mmseg에서 사용하기 위해서 변환하는 과정을 거쳐줘야 합니다.
To use other repositories' pre-trained models, it is necessary to convert keys.
We provide a script swin2mmseg.py in the tools directory to convert the key of models from the official repo to MMSegmentation style.

아래와 같이 사전학습된 weight를 mmseg에 맞는 weight로 바꿔서 저장해줘야 합니다. 
저장하는 변환된 weights의 경로는 `/opt/ml/input/data/pretrain` 로 설정해놨습니다.
```
python tools/model_converters/swin2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH}
```
이렇게 변환해야하는 모델들의 경우 mmsegmentation/tools/model_converters에 들어가서 목록을 확인하실 수 있습니다. 아래는 Swin L에서 사용할 경우의 예시입니다.
**Swin-L**
```
python /opt/ml/input/level2-semantic-segmentation-level2-cv-17/mmseg/mmsegmentation/tools/model_converters/swin2mmseg.py https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth /opt/ml/input/data/pretrain/swin_large_patch4_window12_384_22k.pth
```


This script convert model from PRETRAIN_PATH and store the converted model in STORE_PATH.
(To supply zip format, need pytorch version >= 1.7.0, Use other environment to use)



## Train
### train usage
```
usage: train.py [-h] [--work-dir WORK_DIR] [--load-from LOAD_FROM] [--resume-from RESUME_FROM]
                [--no-validate] [--debug] [--seed SEED] [--tags TAGS [TAGS ...]]
                [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]] [--auto-resume]
                config

Train a segmentor

positional arguments:
  config                test config file path. (EX) pspnet.py

optional arguments:
  -h, --help            show this help message and exit
  --work-dir WORK_DIR   the dir to save logs and models
  --load-from LOAD_FROM
                        the checkpoint file to load weights from, If you are using some checkpoint as
                        pretrain, you should use load_from.
  --resume-from RESUME_FROM
                        the checkpoint file to resume from, When training is interrupted somehow,
                        resume_from should be used to resume training.
  --no-validate         whether not to evaluate the checkpoint during training
  --debug               Debug mode which do not run wandb and epoch 2 
  --seed SEED           random seed
  --tags TAGS [TAGS ...]
                        record your experiment speical keywords into tags list--tags batch_size=16
                        swin_cascasdedont use white space in specific tag
  --cfg-options CFG_OPTIONS [CFG_OPTIONS ...]
                        override some settings in the used config, the key-value pair in xxx=yyy format will
                        be merged into config file. If the value to be overwritten is a list, it should be
                        like key="[a,b]" or key=a,b It also allows nested list/tuple values, e.g.
                        key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white
                        space is allowed.
  --auto-resume         resume from the latest checkpoint automatically. If you want to auto resume with
                        latest checkpoint, use this optionFine latest checkpoint at
                        cfg.work_dirhttps://github.com/open-
                        mmlab/mmdetection/blob/master/mmdet/apis/train.py#L249

```

the key options (Korean)

- config : command에서 반드시 옵션으로 사용하시는 total train config file의 파일명을 넣어줘야 합니다. ( ex-pspnet.py )
- no-validate : 해당 optoin을 주시면, validation을 수행하지 않습니다. 따라서, all dataset으로 학습시키고 싶으시다면, 해당 옵션을 주시면 전체 데이터셋으로 학습하도록 로직이 짜여져 있습니다. ( default는 즉 옵션을 주지 않는다면 validatoin 수행)
- tags : wandb에 로깅할 때, 추가적인 특이사항들을 적어두기 위해 tag설정을 해줍니다. 키워드들을 넣어주시면 됩니다. 예시는 help에 적혀 있습니다. ex) python train.py --tags batch_size=16 swin_cascade
- load-from : 학습된 segmentor의 weights를 load해주고 싶다면 load해주시면 됩니다.
- resume-from : 학습을 중간에 중단해 저장된 체크포인트로부터 다시 학습을 재개하고 싶을 때 사용하시면 됩니다.
- * load와 resume의 차이점은 load는 fine-tunning처럼 학습되는 방식이라면, resume은 저장된 체크포인트 상태에서 바로 다시 재개하는 방식입니다.
- debug : 디버그 모드를 실행시키시면, wandb도 실행되지 않고, max epoch이 2로 설정되어 valid와 train에서 의도대로 동작하는지를 간단히 확인할 수 있습니다. 

### options & cfg-options usage
해당 사용 내용은 help에 자세하게 적혀 있습니다. 간단히 적자면 cfg dict의 값들에 접근해서 config의 옵션을 command line에서 바꿔주기 위한 설정들입니다. 

### Train comand 
- basic  
`python train.py pspnet.py
- more example  
`python train.py pspnet.py  --cfg-options checkpoint_config.max_keep_ckpts=3 runner.max_epochs=30`

## Test
### test usage

```
usage: test.py [-h] [--aug-test] [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]] [--remark REMARK] config checkpoint

mmseg test (and eval) a model

positional arguments:
  config                test config file path. (EX) pspnet.py
  checkpoint            checkpoint file. (EX) exp9/epoch_45.pth

optional arguments:
  -h, --help            show this help message and exit
  --aug-test            Use Flip and Multi scale aug
  --cfg-options CFG_OPTIONS [CFG_OPTIONS ...]
                        override some settings in the used config, the key-value pair in xxx=yyy format will
                        be merged into config file. If the value to be overwritten is a list, it should be
                        like key="[a,b]" or key=a,b It also allows nested list/tuple values, e.g.
                        key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white
                        space is allowed.
  --remark REMARK       uniqueness of submission file when testing (EX) TTA
```

### the key options (Korean)

- config : command에서 반드시 옵션으로 사용하시는 total train config file의 파일명을 넣어줘야 합니다. ( ex-pspnet.py)
- checkpoint : 학습시키면서 생기는 detector의 weigth의 경로를 주셔야 합니다. 이 때 work_dirs에 있는 exp명과 epoch{num}.pth 의 경로를 넣어주시면 됩니다. ( ex- exp9/epoch_45.pth )
- aug-test : TTA를 수행하고 싶다면 해당 옵션을 주시면 됩니다. 추가적으로 ratio를 바꾸고 싶다면, 직접 aug-test에서 바꿔주셔야 합니다.

### options & cfg-options usage
해당 사용 내용은 help에 자세하게 적혀 있습니다. 간단히 적자면 cfg dict의 값들에 접근해서 config의 옵션을 command line에서 바꿔주기 위한 설정들입니다. 

### Test command
- basic  
`python test.py pspnet.py exp9/epoch_45.pth`
- more example  
`python test.py pspnet.py exp9/epoch_45.pth --cfg-options data.test.piepline.1.img_scale="(224,224)" `