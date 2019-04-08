import h5py
import pandas as pd
import os
from os.path import join
import sys
sys.path.append('/data/nlp')
from datasets import *
from config import default as config

type = 'train'
with h5py.File(join(config.wdir, type + '_encodings.h5')) as f:
    encodings = np.array(f['encodings'])
dir = join(config.wdir, 'csv', type)
if not os.path.exists(dir):
    os.makedirs(dir)
with open(join(dir, 'encodings.csv'), 'w+') as f:
    for enc in encodings:
        f.write(','.join([str(v) for v in enc]) + ',0\n')
