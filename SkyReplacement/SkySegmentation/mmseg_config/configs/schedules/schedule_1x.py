optimizer = dict(
    type='AdamW',
    lr=0.00006,
    weight_decay=0.1,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# lr_config = dict(
#         policy='CosineRestart',
#         warmup='exp',
#         periods = [1500,4500,10500,22500],
#         restart_weights = [1,0.5,0.25,0.1],
#         warmup_iters=100, 
#         warmup_ratio=0.01,
#         min_lr=0
#     )
# 만약 위에거를 쓴다면 linear도 갠찮아보이고, periods를 뒤에가 길게가 아닌 앞에가 길게 해야할 듯
lr_config = dict(
        policy='CosineAnnealing',
        warmup='linear',
        warmup_iters=100, 
        warmup_ratio=0.01,
        min_lr=1e-07,
    )
# lr_config = dict(policy='poly', power=0.9, min_lr=0, by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=80)