_base_ = [
    '../models/upernet_swin.py', '../dataset/ade20k.py',
    '../default_runtime.py', '../schedules/schedule_1x.py'
]
# model=dict(
#     decode_head=dict(
#         sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)) )
