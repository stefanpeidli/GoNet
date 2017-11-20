# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 11:08:57 2017

@author: Stefan Peidli

Tags: sgf, converter, board matrix as vector

This Script converts a sgf file into a board matrix (alligne into a vector)

"""

import sgf
import numpy as np

filedirectory="C:/Users/Stefan/Documents/GO-Games Dataset/dgs/game_2981.sgf"
with open(filedirectory) as f:
    collection = sgf.parse(f.read())
    
n=9
game=collection[0] #some collections have more games than one, I ignore them for now
nod=game.nodes
moves=[0]*(len(nod)-1) # the node at 0 is the game info (i think)
player=[0]*(len(nod)-1) # we need to keep track of who played that move, B=-1, W=+1
datam=np.zeros((n,n))
for i in range(1,len(nod)):
    moves[i-1]=list(nod[i].properties.values())[0][0] #magic
    player[i-1]=int(((ord(list(nod[i].properties.keys())[0][0])-66)/21-0.5)*2) # even more magic
    m=moves[i-1]
    if len(m)==2:
        datam[ord(m[0]) - 97,ord(m[1]) - 97]=player[i-1]
data=datam.ravel()
        
print(nod[0].properties)
