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

#---------------------------------------------------------

class TrainingData:
    #standard initialize with boardsize 9
    def __init__(self):
        self.n = 9
        self.board = Board(self.n)
        self.dic = defaultdict(np.ndarray)

    #help method for converting a vector containing a move to the corresponding coord tuple
    def toCoords(self, vector):
        if np.linalg.norm(vector) == 1:
            for entry in range(len(vector)):
                if vector[entry] == 1:
                    return entry % 9, int(entry / 9)
        else: return "Error"

    def importTrainingData(self, folder, von, bis):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        filedir = dir_path + "/" + folder + "/"
        gameIDs = []  # stores the suffix
        for i in range(von, bis):
            currfile = filedir + "game_" + str(i) + ".sgf"
            if os.path.exists(currfile):
                gameIDs = np.append(gameIDs, i)
                self.importSingleFile(currfile)

    def importSingleFile(self, currfile):
        with open(currfile) as f:
            collection = sgf.parse(f.read())

        game = collection[0]  # some collections have more games than one, I ignore them for now
        node = game.nodes
        moves = [0] * (len(node) - 1)  # the node at 0 is the game info (i think)
        player = [0] * (len(node) - 1)  # we need to keep track of who played that move, B=-1, W=+1
        boards = [0] * len(node)
        boards[0] = np.zeros(self.n * self.n, dtype=np.int32)  # first board of the game is always empty board
        boardMatrix = np.zeros((self.n, self.n), dtype=np.int32)
        for i in range(1, len(node)):
            # first we extract the next move
            moves[i - 1] = list(node[i].properties.values())[0][0]
            player[i - 1] = int(((ord(list(node[i].properties.keys())[0][0]) - 66) / 21 - 0.5) * 2)
            m = moves[i - 1]
            if len(m) > 0 and type(m) is str:
                if 97+self.n > ord(m[0]) > 97:  # there are some corrupted files with e.g. "54" as entry, (game 2981)
                    boardMatrix[ord(m[1]) - 97, ord(m[0]) - 97] = player[i - 1]
            boards[i] = boardMatrix.flatten()

            self.addToDict(boards[i - 1], boardMatrix, player[i - 1])

            # now we play the move on the board and see if we have to remove stones
            coords = self.toCoords(np.absolute(boards[i] - boards[i - 1]))
            if type(coords) is not str:
                self.board.play_stone(coords[1],coords[0],player[i-1])
                boards[i] = self.board.vertices.flatten()
                boardMatrix = self.board.vertices

    #help method: adds tuple (board,move) to dictionary for every possible rotation
    def addToDict(self, prevBoardVector, currBoardVector, player):
        prevBoardMatrix = prevBoardVector.reshape((9, 9))
        currPrevPair = (currBoardVector, prevBoardMatrix)
        flippedCurrPrevPair = (np.flip(currPrevPair[0], 1), np.flip(currPrevPair[1], 1))
        symmats = [currPrevPair]
        if not np.array_equal(currPrevPair[0], flippedCurrPrevPair[0]):
            symmats.append(flippedCurrPrevPair)
        for i in range(3):
            currPrevPair = (np.rot90(currPrevPair[0]), np.rot90(currPrevPair[1]))
            flippedCurrPrevPair = (np.rot90(flippedCurrPrevPair[0]), np.rot90(flippedCurrPrevPair[1]))
            for i in range(len(symmats)):
                if np.array_equal(currPrevPair[0], symmats[i][0]):
                    break
                elif i == len(symmats) - 1:
                    symmats.append(currPrevPair)
            for i in range(len(symmats)):
                if np.array_equal(flippedCurrPrevPair[0], symmats[i][0]):
                    break
                elif i == len(symmats) - 1:
                    symmats.append(flippedCurrPrevPair)

        for rotatedPair in symmats:
            currBoardVector = rotatedPair[0].flatten()
            prevBoardVector = rotatedPair[1].flatten()
            if player == -1:  # Trainieren das Netzwerk nur für Spieler Schwarz. wenn weiß: Flip colors B=-1, W=+1
                if str(prevBoardVector) in self.dic:
                    self.dic[str(prevBoardVector)] += np.absolute(currBoardVector - prevBoardVector)
                else:
                    self.dic[str(prevBoardVector)] = np.absolute(currBoardVector - prevBoardVector)
            else:
                invPrevBoardVector = np.zeros(9 * 9, dtype=np.int32)
                for count in range(len(prevBoardVector)):
                    if prevBoardVector[count] != 0:
                        invPrevBoardVector[count] = -1 * prevBoardVector[count]
                    else:
                        invPrevBoardVector[count] = 0
                if str(invPrevBoardVector) in self.dic:
                    self.dic[str(invPrevBoardVector)] += np.absolute(currBoardVector - prevBoardVector)
                else:
                    self.dic[str(invPrevBoardVector)] = np.absolute(currBoardVector - prevBoardVector)

# end class TrainingData

# run
t = TrainingData()
t.importTrainingData("dgs",1,1000)
#print(dic)
print("\n")
for entry in t.dic:
       print ('\n', np.matrix(entry).reshape((9,9)), '\n', t.dic[entry].reshape((9,9)), '\n')
       #print('\n', entry, '\n', t.dic[entry], '\n')
#print(t.dic[str(np.zeros(9*9,dtype=np.int32))].reshape((9,9)))