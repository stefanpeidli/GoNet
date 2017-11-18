# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:27:16 2017

@author: Stefan Peidli

"""
import sgf
import numpy as np
import os.path

def importsgf(file):
    if os.path.exists(file):
        with open(file) as f:
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
            if len(m)==2 and type(m) is str:
                if ord(m[0]) - 97 > 0: #there are some corrupted files with e.g. "54" as entry, (game 2981)
                    datam[ord(m[0]) - 97,ord(m[1]) - 97]=player[i-1]
        return datam.flatten()
    else:
        return "no such file found"


# Now import them all up to game 1000

n=9
mg=50000
filedir="C:/Users/Stefan/Documents/GO-Games Dataset/dgs/" #format: "+game_NUMBER.sgf"
gameslist=np.zeros((mg,n*n))
gameIDs=[]
for i in range(0,mg):
    currfile=filedir+"game_"+str(i)+".sgf"
    imp=importsgf(currfile)
    if type(imp) is not str:
        gameslist[i,:]=imp
        gameIDs=np.append(gameIDs,i)

gameslist=gameslist[gameIDs.astype(int),:]
    
    
    
    



