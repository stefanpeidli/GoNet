# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:27:16 2017

@author: Stefan Peidli

This script reads sgf files from a file directory, converts them into boards (represented by n*n-dim vectors) and saves them all into a big list called gameslist.

"""
import sgf
import numpy as np
import os
from collections import defaultdict

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
            moves[i-1]=list(nod[i].properties.values())[0][0]
            player[i-1]=int(((ord(list(nod[i].properties.keys())[0][0])-66)/21-0.5)*2)
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

def stefantest():
    n=9 #board size
    mg=50 #maximal game suffix "NUMBER" to be checked (first game is game_4.sgf)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filedir=dir_path+"/dgs/"
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


def importTrainingData(folder):
    n = 9  # board size
    mg = 50  # maximal game suffix "NUMBER" to be checked (first game is game_4.sgf)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filedir = dir_path + "/" + folder + "/"
    dic = defaultdict(np.ndarray)
    gameIDs = []  # stores the suffix
    for i in range(0, mg):
        currfile = filedir + "game_" + str(i) + ".sgf"
        if type(dic) is not str:
            gameIDs = np.append(gameIDs, i)

        if os.path.exists(currfile):
            with open(currfile) as f:
                collection = sgf.parse(f.read())

            game = collection[0]  # some collections have more games than one, I ignore them for now
            nod = game.nodes
            moves = [0] * (len(nod) - 1)  # the node at 0 is the game info (i think)
            player = [0] * (len(nod) - 1)  # we need to keep track of who played that move, B=-1, W=+1
            data = [0] * len(nod)
            data[0] = np.zeros(n * n)  # first board of the game is always empty board
            datam = np.zeros((n, n))
            for i in range(1, len(nod)):
                moves[i - 1] = list(nod[i].properties.values())[0][0]
                player[i - 1] = int(((ord(list(nod[i].properties.keys())[0][0]) - 66) / 21 - 0.5) * 2)
                m = moves[i - 1]
                if len(m) == 2 and type(m) is str:
                    if ord(m[0]) - 97 > 0:  # there are some corrupted files with e.g. "54" as entry, (game 2981)
                        datam[ord(m[0]) - 97, ord(m[1]) - 97] = player[i - 1]
                data[i] = datam.flatten()

                if player[i-1] == 1:
                    if str(data[i-1]) in dic:
                        dic[str(data[i-1])] += np.absolute(data[i] - data[i-1])
                    else:
                        dic[str(data[i-1])] = np.absolute(data[i] - data[i-1])
                else:
                    data_i_1 = np.zeros(n*n)
                    for count in range(len(data[i-1])):
                        if data[i-1][count] != 0:
                            data_i_1[count] = -1 * data[i-1][count]
                        else:
                            data_i_1[count] = 0
                    if str(data_i_1) in dic:
                        dic[str(data_i_1)] += np.absolute(data[i] - data[i-1])
                    else:
                        dic[str(data_i_1)] = np.absolute(data[i] - data[i-1])

    return dic




#dic = importTrainingData()
#print(dic)