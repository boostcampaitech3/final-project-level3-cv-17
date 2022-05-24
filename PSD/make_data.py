import os
import random
from shutil import copyfile
from utils import make_directory

data_list = [
    'Crawling',
    'RESIDE_SOTS_OUT',
    'Hidden',
    'MRFID'
]

sample_num = 5

for data in data_list:

    if data == 'Crawling':
        original_path = '../data/Crawling/hazy'
        sample_path = '../data/sample/Crawling'
    elif data == 'RESIDE_SOTS_OUT':
        original_path = '../data/RESIDE_SOTS_OUT/hazy'
        sample_path = '../data/sample/RESIDE_SOTS_OUT'
    elif data == 'Hidden':
        original_path = '../data/Hidden/hazy'
        sample_path = '../data/sample/Hidden'
    elif data == 'MRFID':
        original_path = '../data/MRFID/hazy'
        sample_path = '../data/sample/MRFID'
    else:
        ValueError

    make_directory(sample_path)

    data_sample = random.sample(os.listdir(original_path), sample_num)

    for s_data in data_sample:
        copyfile(os.path.join(original_path, s_data), os.path.join(sample_path, s_data)) 