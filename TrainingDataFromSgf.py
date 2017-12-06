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
from Hashable import Hashable
import pandas as pd
import time


def importsgf_old(file):
    if os.path.exists(file):
        with open(file) as f:
            collection = sgf.parse(f.read())

        n = 9
        game = collection[0]  # some collections have more games than one, I ignore them for now
        nod = game.nodes
        moves = [0] * (len(nod) - 1)  # the node at 0 is the game info (i think)
        player = [0] * (len(nod) - 1)  # we need to keep track of who played that move, B=-1, W=+1
        datam = np.zeros((n, n))
        for i in range(1, len(nod)):
            moves[i - 1] = list(nod[i].properties.values())[0][0]  # magic
            player[i - 1] = int(((ord(list(nod[i].properties.keys())[0][0]) - 66) / 21 - 0.5) * 2)  # even more magic
            m = moves[i - 1]
            if len(m) == 2 and type(m) is str:
                if ord(m[0]) - 97 > 0:  # there are some corrupted files with e.g. "54" as entry, (game 2981)
                    datam[ord(m[0]) - 97, ord(m[1]) - 97] = player[i - 1]
        return datam.flatten()
    else:
        return "no such file found"


def importsgf(file):
    if os.path.exists(file):
        with open(file) as f:
            collection = sgf.parse(f.read())

        n = 9
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
        return data
    else:
        return "no such file found"


# print(importsgf("C:/Users/Stefan/Documents/GO-Games Dataset/dgs/game_4.sgf"))

# Now import them all up to game 1000

def stefantest():
    n = 9  # board size
    mg = 50  # maximal game suffix "NUMBER" to be checked (first game is game_4.sgf)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filedir = dir_path + "/dgs/"
    gameslist = []  # stores the games
    gameIDs = []  # stores the suffix
    for i in range(0, mg):
        currfile = filedir + "game_" + str(i) + ".sgf"
        imp = importsgf(currfile)
        if type(imp) is not str:
            gameslist.append(imp)
            # gameIDs=np.append(gameIDs,i)
            gameIDs = np.append(gameIDs, i)


# by now, gameslist is a list of games that contain Boards

# ---------------------------------------------------------

class TrainingData:
    # standard initialize with boardsize 9
    def __init__(self, folder=None, id_list=range(1000)):
        self.n = 9
        self.board = Board(self.n)
        self.dic = defaultdict(np.ndarray)
        self.test = 0
        if folder is not None:
            self.importTrainingData(folder, id_list)

    # help method for converting a vector containing a move to the corresponding coord tuple
    def toCoords(self, vector):
        vector = vector.flatten()
        if np.linalg.norm(vector) == 1:
            for entry in range(len(vector)):
                if vector[entry] == 1:
                    return entry % 9, int(entry / 9)
        else:
            return None

    def importTrainingData(self, folder, id_list=range(1000)):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        #  print(dir_path + "\ + 'Training_Sets' + '\' + id_list + '.xsl')
        if type(id_list) is str:
            id_list_prepped = pd.read_excel(dir_path + '/' + 'Training_Sets' + '/' + id_list + '.xlsx')[
                "game_id"].values.tolist()
        else:
            id_list_prepped = id_list
        filedir = dir_path + "/" + folder + "/"
        gameIDs = []  # stores the suffix
        for i in id_list_prepped:
            currfile = filedir + "game_" + str(i) + ".sgf"
            if os.path.exists(currfile):
                gameIDs = np.append(gameIDs, i)
                self.importSingleFile(currfile)

    def importSingleFile(self, currfile):
        with open(currfile) as f:
            collection = sgf.parse(f.read())
        self.board.clear()
        # some collections have more games than one, I ignore them for now
        node = collection[0].nodes
        # first board of the game is always empty board
        prevBoardMatrix = np.zeros((self.n, self.n), dtype=np.int32)
        currBoardMatrix = np.zeros((self.n, self.n), dtype=np.int32)

        for i in range(1, len(node)):
            # first we extract the next move, e.g. 'bb'
            move = list(node[i].properties.values())[0][0]
            # we need to keep track of who played that move, B=-1, W=+1
            stone = int(((ord(list(node[i].properties.keys())[0][0]) - 66) / 21 - 0.5) * 2)
            # if file says bs, then stop reading file
            if not (stone == 1 or stone == -1): return
            if not (len(move) > 0 and type(move) is str and 97 + self.n > ord(move[0]) > 97): return

            currBoardMatrix[ord(move[1]) - 97, ord(move[0]) - 97] = stone

            self.addToDict(prevBoardMatrix, currBoardMatrix, stone)

            # now we play the move on the board and see if we have to remove stones
            coords = self.toCoords(np.absolute(currBoardMatrix - prevBoardMatrix))
            if coords is not None:
                self.board.play_stone(coords[1], coords[0], stone)
                prevBoardMatrix = np.copy(self.board.vertices)
                currBoardMatrix = np.copy(self.board.vertices)

                # help method: adds tuple (board,move) to dictionary for every possible rotation

    def addToDict(self, prevBoardMatrix, currBoardMatrix, player):
        currPrevPair = (currBoardMatrix, prevBoardMatrix)
        flippedCurrPrevPair = (np.flip(currPrevPair[0], 1), np.flip(currPrevPair[1], 1))
        symmats = [currPrevPair, flippedCurrPrevPair]
        for i in range(3):
            currPrevPair = (np.rot90(currPrevPair[0]), np.rot90(currPrevPair[1]))
            flippedCurrPrevPair = (np.rot90(flippedCurrPrevPair[0]), np.rot90(flippedCurrPrevPair[1]))
            symmats.append(currPrevPair)
            symmats.append(flippedCurrPrevPair)

        for rotatedPair in symmats:
            currBoardVector = rotatedPair[0].flatten()
            prevBoardVector = rotatedPair[1].flatten()
            if player == -1:  # Trainieren das Netzwerk nur für Spieler Schwarz. wenn weiß: Flip colors B=-1, W=+1
                if Hashable(prevBoardVector) in self.dic:
                    self.dic[Hashable(prevBoardVector)] += np.absolute(currBoardVector - prevBoardVector)
                else:
                    self.dic[Hashable(prevBoardVector)] = np.absolute(currBoardVector - prevBoardVector)
            else:
                invPrevBoardVector = np.zeros(9 * 9, dtype=np.int32)
                for count in range(len(prevBoardVector)):
                    if prevBoardVector[count] != 0:
                        invPrevBoardVector[count] = -1 * prevBoardVector[count]
                    else:
                        invPrevBoardVector[count] = 0
                if Hashable(invPrevBoardVector) in self.dic:
                    self.dic[Hashable(invPrevBoardVector)] += np.absolute(currBoardVector - prevBoardVector)
                else:
                    self.dic[Hashable(invPrevBoardVector)] = np.absolute(currBoardVector - prevBoardVector)


# end class TrainingData

class TrainingDataSgf:
    # standard initialize with boardsize 9
    def __init__(self, folder=None, id_list=range(1000)):
        self.n = 9
        self.board = Board(self.n)
        self.dic = defaultdict(np.ndarray)
        if folder is not None:
            self.importTrainingData(folder, id_list)

    # help method for converting a vector containing a move to the corresponding coord tuple
    def toCoords(self, vector):
        vector = vector.flatten()
        if np.linalg.norm(vector) == 1:
            for entry in range(len(vector)):
                if vector[entry] == 1:
                    return entry % 9, int(entry / 9)
        else:
            return None

    def importTrainingData(self, folder, id_list=range(1000)):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        #  print(dir_path + "\ + 'Training_Sets' + '\' + id_list + '.xsl')
        if type(id_list) is str:
            id_list_prepped = pd.read_excel(dir_path + '/' + 'Training_Sets' + '/' + id_list + '.xlsx')[
                "game_id"].values.tolist()
        else:
            id_list_prepped = id_list
        filedir = dir_path + "/" + folder + "/"
        gameIDs = []  # stores the suffix
        for i in id_list_prepped:
            currfile = filedir + "game_" + str(i) + ".sgf"
            if os.path.exists(currfile):
                gameIDs = np.append(gameIDs, i)
                self.importSingleFile(currfile)

    def importSingleFile(self, currfile):
        with open(currfile, encoding="Latin-1") as f:
            movelistWithAB = f.read().split(';')[1:]
            movelist = movelistWithAB[1:]
        self.board.clear()

        # first board of the game is always empty board
        prevBoardMatrix = np.zeros((self.n, self.n), dtype=np.int32)
        currBoardMatrix = np.zeros((self.n, self.n), dtype=np.int32)
        # NOT always empty: add black handycap stones
        for line in movelistWithAB[0].split('\n'):
            if line.startswith("AB"):
                handycapmoves = []
                m = line.split('[')
                for i in range(1, len(m) - 1):
                    if not m[i].startswith('P'):
                        handycapmoves.append('B[' + m[i][0] + m[i][1] + ']')
                movelist = handycapmoves + movelist

        # some corrupt files have more ';" at beginning
        if movelist[0].startswith('&') or movelist[0].startswith(' ') or movelist[0].startswith('d') \
                or movelist[0].startswith('(') or movelist[0].startswith('i') or movelist[0].startswith('o') \
                or len(movelist[0]) > 20:
            for m in movelist:
                if m.startswith('&') or m.startswith(' ') or m.startswith('d') \
                        or m.startswith('(') or m.startswith('i') or m.startswith('o') \
                        or len(m) > 20:
                    movelist = movelist[1:]

        for i in range(0, len(movelist)):
            # we need to keep track of who played that move, B=-1, W=+1
            stoneColor = movelist[i].split('[')[0]
            if stoneColor == 'B':
                stone = -1
            elif stoneColor == 'W':
                stone = 1
            # game finished
            elif stoneColor == 'C':
                return
            # PL tells whose turn it is to play in move setup situation ??
            elif stoneColor == "PL":
                return
            # Comment in end
            elif len(stoneColor) > 2:
                return
            # MoveNumber added at before move
            elif stoneColor == "MN":
                movelist[i] = movelist[i].split(']')[1]
                stoneColor = movelist[i].split('[')[0]
                if stoneColor == 'B':
                    stone = -1
                elif stoneColor == 'W':
                    stone = 1
            else:
                return

            # now we extract the next move, e.g. 'bb'
            move = movelist[i].split('[')[1].split(']')[0]

            # print(stoneColor, "plays", move)

            # If not Pass
            if len(move) > 0:
                currBoardMatrix[ord(move[1]) - 97, ord(move[0]) - 97] = stone

            self.addToDict(prevBoardMatrix, currBoardMatrix, stone)

            # now we play the move on the board and see if we have to remove stones
            coords = self.toCoords(np.absolute(currBoardMatrix - prevBoardMatrix))
            if coords is not None:
                self.board.play_stone(coords[1], coords[0], stone)
                prevBoardMatrix = np.copy(self.board.vertices)
                currBoardMatrix = np.copy(self.board.vertices)

                # help method: adds tuple (board,move) to dictionary for every possible rotation

    def addToDict(self, prevBoardMatrix, currBoardMatrix, player):
        currPrevPair = (currBoardMatrix, prevBoardMatrix)
        flippedCurrPrevPair = (np.flip(currPrevPair[0], 1), np.flip(currPrevPair[1], 1))
        symmats = [currPrevPair, flippedCurrPrevPair]
        for i in range(3):
            currPrevPair = (np.rot90(currPrevPair[0]), np.rot90(currPrevPair[1]))
            flippedCurrPrevPair = (np.rot90(flippedCurrPrevPair[0]), np.rot90(flippedCurrPrevPair[1]))
            symmats.append(currPrevPair)
            symmats.append(flippedCurrPrevPair)

        for rotatedPair in symmats:
            currBoardVector = rotatedPair[0].flatten()
            prevBoardVector = rotatedPair[1].flatten()
            if player == -1:  # Trainieren das Netzwerk nur für Spieler Schwarz. wenn weiß: Flip colors B=-1, W=+1
                if Hashable(prevBoardVector) in self.dic:
                    self.dic[Hashable(prevBoardVector)] += np.absolute(currBoardVector - prevBoardVector)
                else:
                    self.dic[Hashable(prevBoardVector)] = np.absolute(currBoardVector - prevBoardVector)
            else:
                invPrevBoardVector = np.zeros(9 * 9, dtype=np.int32)
                for count in range(len(prevBoardVector)):
                    if prevBoardVector[count] != 0:
                        invPrevBoardVector[count] = -1 * prevBoardVector[count]
                    else:
                        invPrevBoardVector[count] = 0
                if Hashable(invPrevBoardVector) in self.dic:
                    self.dic[Hashable(invPrevBoardVector)] += np.absolute(currBoardVector - prevBoardVector)
                else:
                    self.dic[Hashable(invPrevBoardVector)] = np.absolute(currBoardVector - prevBoardVector)


# end class TrainingDataSgf

# run
def test():
    t = TrainingData("dgs", range(10000))

    # print complete dictionary, don't if dic is big ;)
    # for entry in t.dic:
    #      testdata = Hashable.unwrap(entry)
    #      targ = t.dic[entry].reshape(9*9)
    #      print('\n', '\n', Hashable.unwrap(entry), '\n', t.dic[entry].reshape((9,9)), '\n')

    # print cumulated move distribution for empty board
    zeroMatrix = t.dic[Hashable(np.zeros(t.n * t.n, dtype=np.int32))]
    print('\n', zeroMatrix.reshape((9, 9)), '\n')
    print(np.sum(zeroMatrix))

    # print cumulated move distributions for boards with exactly one stone
    secondMoveDist = np.zeros(9 * 9, dtype=np.int32)
    for entry in t.dic:
        thisis = Hashable.unwrap(entry)
        if np.sum(np.absolute(thisis)) == 1:
            secondMoveDist += t.dic[entry]
    print(secondMoveDist.reshape((9, 9)))
    print(np.sum(secondMoveDist))


# test()

def test2():
    start = time.clock()
    l = []
    t = TrainingData("dgs", range(0, 100))
    length = 0
    for entry in t.dic:
        length += 1
    l.append(length)
    print(l)
    print(time.clock() - start)
    print(t.test)


# test2()

def test3():
    start = time.clock()
    t = TrainingDataSgf("dgs", range(300000))

    zeroMatrix = t.dic[Hashable(np.zeros(t.n * t.n, dtype=np.int32))]
    print('\n', zeroMatrix.reshape((9, 9)), '\n')
    print(np.sum(zeroMatrix))

    # print cumulated move distributions for boards with exactly one stone
    secondMoveDist = np.zeros(9 * 9, dtype=np.int32)
    for entry in t.dic:
        thisis = Hashable.unwrap(entry)
        if np.sum(np.absolute(thisis)) == 1:
            secondMoveDist += t.dic[entry]
    print(secondMoveDist.reshape((9, 9)))
    print(np.sum(secondMoveDist))
    print("\nTime " + str(time.clock() - start))
# test3()