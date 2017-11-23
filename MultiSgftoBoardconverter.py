# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:27:16 2017

@author: Stefan Peidli

This script reads sgf files from a file directory, converts them into boards (represented by n*n-dim vectors) and saves them all into a big list called gameslist.

"""
import sgf
import numpy as np
import os.path

def importsgf_old(file):
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
    
def importsgf(file):
    if os.path.exists(file):
        with open(file) as f:
            collection = sgf.parse(f.read())
            
        n=9
        game=collection[0] #some collections have more games than one, I ignore them for now
        nod=game.nodes
        moves=[0]*(len(nod)-1) # the node at 0 is the game info (i think)
        player=[0]*(len(nod)-1) # we need to keep track of who played that move, B=-1, W=+1
        data=[0]*len(nod)
        data[0]=np.zeros(n*n)#first board of the game is always empty board
        datam=np.zeros((n,n))
        for i in range(1,len(nod)):
            moves[i-1]=list(nod[i].properties.values())[0][0] #magic
            player[i-1]=int(((ord(list(nod[i].properties.keys())[0][0])-66)/21-0.5)*2) # even more magic
            m=moves[i-1]
            if len(m)==2 and type(m) is str:
                if ord(m[0]) - 97 > 0: #there are some corrupted files with e.g. "54" as entry, (game 2981)
                    datam[ord(m[0]) - 97,ord(m[1]) - 97]=player[i-1]
            data[i]=datam.flatten()
        return data
    else:
        return "no such file found"

#print(importsgf("C:/Users/Stefan/Documents/GO-Games Dataset/dgs/game_4.sgf"))

# Now import them all up to game 1000

n=9 #board size
mg=50000 #maximal game suffix "NUMBER" to be checked (first game is game_4.sgf)
filedir="C:/Users/Stefan/Documents/GO-Games Dataset/dgs/" #format: "+game_NUMBER.sgf", change this to your directory!
gameslist=[] #stores the games
gameIDs=[] #stores the suffix
for i in range(0,mg):
    currfile=filedir+"game_"+str(i)+".sgf"
    imp=importsgf(currfile)
    if type(imp) is not str:
        gameslist.append(imp)
        #gameIDs=np.append(gameIDs,i)
        gameIDs=np.append(gameIDs,i)

#by now, gameslist is a list of games that contain Boards

    
    
    



