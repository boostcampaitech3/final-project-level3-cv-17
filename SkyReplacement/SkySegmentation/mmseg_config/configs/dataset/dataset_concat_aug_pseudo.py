# dataset settings
dataset_type = 'CustomDataset'
img_dir='/opt/ml/input/data/mmseg/img_dir/'
ann_dir= '/opt/ml/input/data/mmseg/ann_dir/'

classes = ("Backgroud","General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
palette =  [[0,0,0], [192,0,128], [0,128,192], [0,128,64], [128,0,0], [64,0,128],
           [64,0,192] ,[192,128,64], [192,192,128], [64,64,128], [128,0,192]]

img_norm_cfg = dict(
    mean = [109.9291, 117.2673, 123.4647] , std = [54.8851, 53.497 , 54.0975], to_rgb=True)
# train_all mean, std


train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(512,512), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='RandomRotate', prob=0.5,degree=(0,359)),
        dict(type='RandomCrop',crop_size=(384,384),cat_max_ratio=0.7),
        dict(type ='PhotoMetricDistortion'),  
        dict(type='RandomCutOut',prob=0.3,n_holes=7, cutout_shape=
                    [
                       (8, 8), (16, 8), (8, 16),
                       (16, 16), (16, 32), (32, 16) ]),

        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

val_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1024, 1024),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=[(512,512)],#[(1024, 1024),(512,512),(1333,800)],
            flip= False,
            flip_direction =  ["horizontal"],
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

train_d=dict(
        type=dataset_type,
        ann_dir=ann_dir + 'train_all_clean',
        img_dir=img_dir + 'train_all_clean',
        classes = classes,
        palette= palette,
        pipeline=train_pipeline)

leak=dict(
        type=dataset_type,
        ann_dir=ann_dir + 'leak',
        img_dir=img_dir + 'leak',
        classes = classes,
        palette= palette,
        pipeline=train_pipeline)

pseudo=dict(
        type=dataset_type,
        ann_dir=ann_dir + 'pseudo_8041',
        img_dir=img_dir + 'pseudo_8041',
        classes = classes,
        palette= palette,
        pipeline=train_pipeline)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train = [
        train_d,
        leak,
        pseudo
    ],
    val=dict(
        type=dataset_type,
        ann_dir=ann_dir + 'val',
        img_dir=img_dir + 'val',
        classes = classes,
        palette= palette,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        img_dir=img_dir+'test' ,
        classes = classes,
        palette= palette,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='mIoU',    
                classwise=True, save_best ='mIoU'
                )