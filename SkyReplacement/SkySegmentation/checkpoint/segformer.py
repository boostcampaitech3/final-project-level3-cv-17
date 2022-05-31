norm_cfg = dict(type='BN', requires_grad=True)
backbone_pretrained = '/opt/ml/input/final-project-level3-cv-17/SkyReplacement/SkySegmentation/mmseg_config/pretrain/swin_base_patch4_window7_224.pth'
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_size=4,
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        mlp_ratio=4,
        norm_cfg=dict(type='LN', requires_grad=True),
        act_cfg=dict(type='GELU'),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/opt/ml/input/final-project-level3-cv-17/SkyReplacement/SkySegmentation/mmseg_config/pretrain/swin_base_patch4_window7_224.pth'
        )),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        channels=768,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=3.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'CustomDataset'
img_dir = '/opt/ml/input/final-project-level3-cv-17/data/skyseg/img_dir/'
ann_dir = '/opt/ml/input/final-project-level3-cv-17/data/skyseg/ann_dir/'
classes = ('BackGroud', 'Sky')
palette = [[0, 0, 0], [255, 255, 255]]
img_norm_cfg = dict(
    mean=[109.9291, 117.2673, 123.4647],
    std=[54.8851, 53.497, 54.0975],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='EdgeAug'),
    dict(type='BlueEmphasis', low=190, prob=0.5),
    dict(type='BlueStretch', prob=0.5),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0),
    dict(
        type='Normalize',
        mean=[109.9291, 117.2673, 123.4647],
        std=[54.8851, 53.497, 54.0975],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
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
            dict(
                type='Normalize',
                mean=[109.9291, 117.2673, 123.4647],
                std=[54.8851, 53.497, 54.0975],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(512, 512)],
        flip=False,
        flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[109.9291, 117.2673, 123.4647],
                std=[54.8851, 53.497, 54.0975],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='CustomDataset',
        ann_dir=
        '/opt/ml/input/final-project-level3-cv-17/data/skyseg/ann_dir/train',
        img_dir=
        '/opt/ml/input/final-project-level3-cv-17/data/skyseg/img_dir/train',
        classes=('BackGroud', 'Sky'),
        palette=[[0, 0, 0], [255, 255, 255]],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='EdgeAug'),
            dict(type='BlueEmphasis', low=190, prob=0.5),
            dict(type='BlueStretch', prob=0.5),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0),
            dict(
                type='Normalize',
                mean=[109.9291, 117.2673, 123.4647],
                std=[54.8851, 53.497, 54.0975],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='CustomDataset',
        ann_dir=
        '/opt/ml/input/final-project-level3-cv-17/data/skyseg/ann_dir/val',
        img_dir=
        '/opt/ml/input/final-project-level3-cv-17/data/skyseg/img_dir/val',
        classes=('BackGroud', 'Sky'),
        palette=[[0, 0, 0], [255, 255, 255]],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[109.9291, 117.2673, 123.4647],
                        std=[54.8851, 53.497, 54.0975],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CustomDataset',
        img_dir=
        '/opt/ml/input/final-project-level3-cv-17/data/skyseg/img_dir/test',
        img_suffix='.png',
        classes=('BackGroud', 'Sky'),
        palette=[[0, 0, 0], [255, 255, 255]],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(512, 512)],
                flip=False,
                flip_direction=['horizontal', 'vertical'],
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[109.9291, 117.2673, 123.4647],
                        std=[54.8851, 53.497, 54.0975],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='mIoU', classwise=True)
checkpoint_config = dict(interval=2, max_keep_ckpts=5)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='final_project_hyo',
                entity='mg_generation',
                group='skysegmentation',
                reinit=True))
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    weight_decay=0.1,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.01,
    min_lr=1e-07)
runner = dict(type='EpochBasedRunner', max_epochs=80)
work_dir = 'work_dirs/exp82'
gpu_ids = [0]
auto_resume = False
