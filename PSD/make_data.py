import os
import random
from shutil import copyfile

data_type = 'MRFID'

if data_type == 'Crawling':
    original_path = '/opt/ml/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors/PSD/data/crawling'
    sample_path = '/opt/ml/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors/PSD/data/sample/crawl_s'
    sample_num = 7
elif data_type == 'SOTS':
    original_path = '/opt/ml/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors/PSD/data/outdoor/hazy'
    sample_path = '/opt/ml/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors/PSD/data/sample/sots_s'
    sample_num = 7
elif data_type == 'baek':
    original_path = '/opt/ml/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors/PSD/data/baek'
    sample_path = '/opt/ml/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors/PSD/data/sample/baek_s'
    sample_num = 3
elif data_type == 'MRFID':
    original_path = '/opt/ml/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors/PSD/data/MRFID/fog'
    sample_path = '/opt/ml/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors/PSD/data/sample/MRFID'
    sample_num = 7
else:
    ValueError

data_sample = random.sample(os.listdir(original_path), sample_num)

for s_data in data_sample:
    copyfile(os.path.join(original_path, s_data), os.path.join(sample_path, s_data)) 