import pandas as pd
import h5py
from config import default as config
import numpy as np
import sys
import os
from os.path import join
from sklearn.metrics import precision_recall_curve

with h5py.File(config.wdir + '/train_encodings.h5') as f:
    encodings = np.array(f['encodings'])
    filenames = [s.decode('utf-8') for s in f['filenames']]
cluster_dim = str(sys.argv[1])
with open(join(config.wdir, 'csv', 'train', 'clusters_' + cluster_dim + '.csv')) as f:
    clusters = np.array(f.read().splitlines(), dtype=int).tolist()

'''sys.path.append('../lvae/scripts/csv_util.py')
encodings = np.array(pd.read_csv('../lvae/model/birds/csv/train_encodings/0', header=None).values, dtype=float)
clusters = np.squeeze(np.array(pd.read_csv('../lvae/model/birds/csv/train_clusters/0', header=None).values, dtype=int))
filenames = np.squeeze(np.array(pd.read_csv('../lvae/model/birds/csv/train_filenames/0', header=None).values))'''

valid = pd.read_csv('/data/nlp/birds/validation.txt', sep='\t', index_col=False, header=None, names=['filename', 'valid'])
valid['filename'] = valid['filename'].str.replace('/data/nlp/birds/train/', '')
df = pd.DataFrame(data={'filename': filenames, 'cluster': clusters})
df = df.set_index('filename').join(valid.set_index('filename'))
df['cluster'] += 1
df['pred'] = df['cluster'].clip(0, 1)

'''largest_cluster = df[df['cluster'] > 0].groupby(['cluster']).count()['valid'].idxmax()
def pred(row):
    if row['cluster'] == largest_cluster:
        return 1
    else:
        return 0
df['pred'] = df.apply(lambda row: pred(row), axis=1)'''

print(df.groupby(['cluster']).mean().sort_values('valid', ascending=True).iloc[0]['valid'])

'''true = list(df['valid'].values)
pred = list(df['pred'].values)
precision, recall, _ = precision_recall_curve(true, pred)
for p, r in zip(precision, recall):
    print(p, r)'''
