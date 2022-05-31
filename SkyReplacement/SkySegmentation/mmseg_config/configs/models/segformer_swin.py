# model settings
norm_cfg = dict(type='BN', requires_grad=True)
backbone_pretrained = '/opt/ml/input/final-project-level3-cv-17/SkyReplacement/SkySegmentation/mmseg_config/pretrain/swin_base_patch4_window7_224.pth'
backbone_norm_cfg = dict(type='LN', requires_grad=True)
# model = dict(
#     backbone=dict(
#         init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
#         embed_dims=128,
#         depths=[2, 2, 18, 2],
#         num_heads=[4, 8, 16, 32],
#         window_size=7,
#         use_abs_pos_embed=False,
#         drop_path_rate=0.3,
#         patch_norm=True),
#     decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=2),
#     auxiliary_head=dict(in_channels=512, num_classes=2))

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
        drop_rate=0.,
        attn_drop_rate=0.,
        mlp_ratio=4,
        norm_cfg=backbone_norm_cfg,
        act_cfg=dict(type='GELU'),
        init_cfg = dict(type="Pretrained",checkpoint=backbone_pretrained)
        ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        channels=768,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=3.0),
            # dict(type='LovaszLoss',loss_type='multi_class',reduction='none', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)]),#avg_non_ignore=True)),
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=512,
    #     in_index=2,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.7)),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
