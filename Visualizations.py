# -*- coding: utf-8 -*-
"""
@author: Stefan Peidli
License: MIT
Tags: Neural Network
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from Filters import apply_filters_by_id as filt

n = 9

test_data = np.round(np.random.uniform(-1, 1, (n, n)), 0)


# Visualization of a matrix on a board representation:
def show_heatmap(data):
    image = data
    image = image.reshape((n, n))  # Reshape things into a nxn grid.
    row_labels = reversed(np.array(range(n))+1)  # fuck Python
    col_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    plt.matshow(image)
    plt.xticks(range(n), col_labels)
    plt.yticks(range(n), row_labels)
    plt.show()


# make a color map of fixed colors
cmap = colors.ListedColormap(['white', '#e8b468', 'black'])
bounds = [-1.5, -0.5, 0.5, 1]
norm = colors.BoundaryNorm(bounds, cmap.N)


def show_filtered_map(data, color, submode=False, maxid=8):
    f = [data]
    for i in range(maxid):
        f.append(np.asanyarray(filt(test_data, color, [i])).reshape((n, n)))
    if not submode:
        for image in f:
            plt.matshow(image)
            plt.show()
    else:
        fig = plt.figure()
        fig.subplots_adjust(left=0.2, wspace=0.6)
        ax = []
        for i in range(maxid):
            ax.append(fig.add_subplot(331+i))
            ax[i].imshow(f[i], interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
            ax[i].set_yticks(range(n))
            ax[i].set_yticklabels(reversed(np.array(range(n))+1))
            ax[i].set_xticks(range(n))
            ax[i].set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
            ax[i].set_title("Filter ID "+str(i+1))
        fig.tight_layout()
        plt.show()


show_filtered_map(test_data, 1, submode=True)
# show_heatmap(test_data)

