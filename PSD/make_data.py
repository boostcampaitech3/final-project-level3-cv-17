import os
import random
from shutil import copyfile
from utils import make_directory


if __name__ == '__main__':
    
    folder_list = [
        'Crawling',
        'Hidden',
        'MRFID',
        'BeDDE',
        'RESIDE_RTTS',
    ]

    sample_num = 5

    for folder_name in folder_list:

        original_path = '../data/' + folder_name + '/hazy'
        sample_path = '../data/sample/' + folder_name

        make_directory(sample_path)

        data_sample = random.sample(os.listdir(original_path), sample_num)

        for s_data in data_sample:
            copyfile(os.path.join(original_path, s_data), os.path.join(sample_path, s_data)) 