# -*- coding: utf-8 -*-
"""
@author: Stefan Peidli
License: MIT
Tags: Neural Network
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
cmap = colors.ListedColormap(['red', 'black', '#e8b468', 'white', 'green', "blue"])
bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
norm = colors.BoundaryNorm(bounds, cmap.N)


def show_filtered_map(data, color, submode=False, maxid=8, grid=True):
    f = [data]
    for i in range(maxid-1):
        f.append(np.asanyarray(filt(test_data, color, [i])).reshape((n, n)))
    f.append(np.asanyarray(filt(test_data, color, [maxid-1])).reshape((n, n)))
    if not submode:
        for image in f:
            plt.matshow(image)
            plt.show()
    else:
        fig = plt.figure()
        fig.subplots_adjust(left=0.2, wspace=0.6)
        ax = []
        subplot_names = ['board', 'eyes', "eyes_create", "captures", "add_liberties", "liberization", "groups",
                         'only own color', "only enemy color"]
        for i in range(maxid+1):
            ax.append(fig.add_subplot(331+i))
            ag = ax[i].imshow(f[i], interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
            ax[i].set_yticks(range(n))
            ax[i].set_yticklabels(reversed(np.array(range(n))+1))
            ax[i].set_xticks(range(n))
            ax[i].set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
            if grid:
                ax[i].set_xticks(np.arange(-.5, n, 1), minor=True)
                ax[i].set_yticks(np.arange(-.5, n, 1), minor=True)
                ax[i].grid(which='minor', color='black', linestyle='-', linewidth=3)
            # ax[i].set_title("Filter ID "+str(i+1))
            ax[i].set_title("Filter " + subplot_names[i])
        fig.tight_layout()
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cb = fig.colorbar(ag, cax=cbar_ax)
        cb.set_ticks([-2, -1, 0, 1, 2, 3])
        cb.set_ticklabels(['<-2', '-1 (black)', '0 (empty)', '1 (white)', 2, '>2'])

        plt.show()


show_filtered_map(test_data, 1, submode=True)
# show_heatmap(test_data)

