_base_ = [
    '../models/segformer_swin.py', '../dataset/dataset.py',
    '../default_runtime.py', '../schedules/schedule_1x.py'
]

# model = dict(

#     decode_head=dict(#in_channels=[64, 128, 320, 512],
#     sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)))

