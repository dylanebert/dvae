import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import pandas as pd
import sys
from PIL import Image
import os
from sklearn.decomposition import PCA
from os.path import join

def plot_encodings(encodings, filenames, clusters=None):
    fig = plt.figure()
    plt.box(on=None)
    plt.grid(alpha=.5, linestyle='-', linewidth=1)
    cmap = ListedColormap(sns.color_palette('husl', 8).as_hex())
    if not clusters == None:
        clusters = np.array(clusters) + 1
        n = np.amax(clusters)
        colors = [cmap(i / float(n)) for i in clusters]
    else:
        colors = [cmap(0)] * len(encodings)
    if encodings.shape[1] > 2:
        encodings = PCA(n_components=2).fit_transform(encodings)
    plt.scatter(encodings[:,0], encodings[:,1], s=5, picker=5, c=colors)
    #plt.xlim([-4, 4])
    #plt.ylim([-4, 4])

    def onpick(event):
        ind = event.ind[0]
        img = Image.open(os.path.join(config.ddir, 'train', filenames[ind]))
        img.show()
    fig.canvas.mpl_connect('pick_event', onpick)

    plt.show()

from config import default as config
with h5py.File(config.wdir + '/train_encodings.h5') as f:
    encodings = np.array(f['encodings'])
    filenames = [s.decode('utf-8') for s in f['filenames']]
cluster_dim = str(sys.argv[1])
with open(join(config.wdir, 'csv', 'train', 'clusters_' + cluster_dim + '.csv')) as f:
    clusters = np.array(f.read().splitlines(), dtype=int).tolist()
plot_encodings(encodings, filenames, clusters=clusters)
