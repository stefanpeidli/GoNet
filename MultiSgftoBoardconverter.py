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
from Board import *

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

def toCoords(data):
    if np.linalg.norm(data) == 1:
        for entry in range(len(data)):
            if data[entry] == 1:
                return entry % 9, int(entry / 9)
    else: return "Error"



dic = defaultdict(np.ndarray)

def addToDict(prevData, currData, player):
    if player == -1:  # Trainieren das Netzwerk nur für Spieler Schwarz. wenn weiß: Flip colors B=-1, W=+1
        if str(prevData) in dic:
            dic[str(prevData)] += np.absolute(currData - prevData)
        else:
            dic[str(prevData)] = np.absolute(currData - prevData)
    else:
        data_i_1 = np.zeros(9 * 9, dtype=np.int32)
        for count in range(len(prevData)):
            if prevData[count] != 0:
                data_i_1[count] = -1 * prevData[count]
            else:
                data_i_1[count] = 0
        if str(data_i_1) in dic:
            dic[str(data_i_1)] += np.absolute(currData - prevData)
        else:
            dic[str(data_i_1)] = np.absolute(currData - prevData)

def addToDictWithSymmetries(prevData, datam, player):
    prevDataTemp = prevData.reshape((9,9))
    temp = (datam, prevDataTemp)
    temp2 = (np.flip(temp[0],1),np.flip(temp[1],1))
    symmats = [temp]
    if not np.array_equal(temp[0],temp2[0]):
        symmats.append(temp2)
    for i in range(3):
        temp = (np.rot90(temp[0]),np.rot90(temp[1]))
        temp2 = (np.rot90(temp2[0]),np.rot90(temp2[1]))
        for i in range(len(symmats)):
            if np.array_equal(temp[0],symmats[i][0]):
                break
            elif i == len(symmats)-1:
                symmats.append(temp)
        for i in range(len(symmats)):
            if np.array_equal(temp2[0],symmats[i][0]):
                break
            elif i == len(symmats)-1:
                symmats.append(temp2)

    for rots in symmats:
        currData = rots[0].flatten()
        prevData = rots[1].flatten()
        if player == -1:  # Trainieren das Netzwerk nur für Spieler Schwarz. wenn weiß: Flip colors B=-1, W=+1
            if str(prevData) in dic:
                dic[str(prevData)] += np.absolute(currData - prevData)
            else:
                dic[str(prevData)] = np.absolute(currData - prevData)
        else:
            data_i_1 = np.zeros(9 * 9, dtype=np.int32)
            for count in range(len(prevData)):
                if prevData[count] != 0:
                    data_i_1[count] = -1 * prevData[count]
                else:
                    data_i_1[count] = 0
            if str(data_i_1) in dic:
                dic[str(data_i_1)] += np.absolute(currData - prevData)
            else:
                dic[str(data_i_1)] = np.absolute(currData - prevData)

def importTrainingData(folder, von, bis):
    n = 9  # board size
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filedir = dir_path + "/" + folder + "/"
    gameIDs = []  # stores the suffix
    board = Board(n)
    for i in range(von, bis):
        board.clear()
        currfile = filedir + "game_" + str(i) + ".sgf"

        if os.path.exists(currfile):
            gameIDs = np.append(gameIDs, i)
            with open(currfile) as f:
                collection = sgf.parse(f.read())

            game = collection[0]  # some collections have more games than one, I ignore them for now
            nod = game.nodes
            moves = [0] * (len(nod) - 1)  # the node at 0 is the game info (i think)
            player = [0] * (len(nod) - 1)  # we need to keep track of who played that move, B=-1, W=+1
            data = [0] * len(nod)
            data[0] = np.zeros(n * n, dtype=np.int32)  # first board of the game is always empty board
            datam = np.zeros((n, n), dtype=np.int32)
            for i in range(1, len(nod)):
                # first we extract the next move
                moves[i - 1] = list(nod[i].properties.values())[0][0]
                player[i - 1] = int(((ord(list(nod[i].properties.keys())[0][0]) - 66) / 21 - 0.5) * 2)
                m = moves[i - 1]
                if len(m) > 0 and type(m) is str:
                    if 97+n > ord(m[0]) > 97:  # there are some corrupted files with e.g. "54" as entry, (game 2981)
                        datam[ord(m[1]) - 97, ord(m[0]) - 97] = player[i - 1]
                data[i] = datam.flatten()

                addToDictWithSymmetries(data[i - 1], datam, player[i-1])
                #addToDict(data[i-1], data[i], player[i-1])

                # now we play the move on the board and see if we have to remove stones
                coords = toCoords(np.absolute(data[i] - data[i-1]))
                if type(coords) is not str:
                    board.play_stone(coords[1],coords[0],player[i-1])
                    data[i] = board.vertices.flatten()
                    datam = board.vertices


importTrainingData("dgs",1,1000)
#print(dic)
print("\n")
#for entry in dic:
       #print ('\n', np.matrix(entry).reshape((9,9)), '\n', dic[entry].reshape((9,9)), '\n')
       #print('\n', entry, '\n', dic[entry], '\n')
print(dic[str(np.zeros(9*9,dtype=np.int32))].reshape((9,9)))