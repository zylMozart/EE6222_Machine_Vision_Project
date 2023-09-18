import os
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

temp=imgs[0][0].cpu()
for i in range(temp.shape[0]):
    for j in range(temp.shape[1]):
        fig, ax = plt.subplots()
        im = ax.imshow(np.array(temp[i][j]))
        fig.tight_layout()
        plt.savefig('tmp/heatmap/{}_{}.png'.format(i,j))