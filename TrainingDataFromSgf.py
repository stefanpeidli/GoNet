# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:27:16 2017

@author: Stefan Peidli

This script reads sgf files from a file directory, converts them into pairs of "current board - next move" and stores
them either in a dictionary or in a sqlite3 database.

"""
import os
from collections import defaultdict
from Board import *
from Filters import *
from Hashable import Hashable
import pandas as pd
import time
import sqlite3
import io

#neccessarry to store arrays in database (from stackOverFlow)
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

class TrainingDataSgfPass:
    # standard initialize with boardsize 9
    def __init__(self, folder=None, id_list=range(1000), dbNameMoves = False, dbNameDist = False):
        self.n = 9
        self.board = Board(self.n)
        self.dic = defaultdict(np.ndarray)
        self.passVector = np.zeros(self.n*self.n + 1, dtype=np.int32)
        self.passVector[self.n*self.n]=1
        self.dbFlagMoves = False
        self.dbFlagDist = False
        if dbNameMoves:
            self.dbFlagMoves = True
            self.dbNameMoves = dbNameMoves
            con = sqlite3.connect(r"DB's/MoveDB's/" + self.dbNameMoves, detect_types=sqlite3.PARSE_DECLTYPES)
            cur = con.cursor()
            cur.execute("create table test (id INTEGER PRIMARY KEY, board array, move array)")
            con.close()
        if dbNameDist:
            self.dbFlagDist = True
            self.dbNameDist = dbNameDist
            con = sqlite3.connect(r"DB's/DistributionDB's/" + self.dbNameDist, detect_types=sqlite3.PARSE_DECLTYPES)
            cur = con.cursor()
            cur.execute("create table test (id INTEGER PRIMARY KEY, board array, distribution array)")
            con.close()

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
        # close db after last call of importSingleFile
        if self.dbFlagMoves:
            con = sqlite3.connect(r"DB's/MoveDB's/" + self.dbNameMoves, detect_types=sqlite3.PARSE_DECLTYPES)
            con.close()
        if self.dbFlagDist:
            con = sqlite3.connect(r"DB's/DistributionDB's/" + self.dbNameDist, detect_types=sqlite3.PARSE_DECLTYPES)
            con.close()


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

            # add pass as move to dic
            else:
                self.addToDict(prevBoardMatrix, currBoardMatrix, stone, passing=True)

    # help method: adds tuple (board,move) to dictionary for every possible rotation
    def addToDict(self, prevBoardMatrix, currBoardMatrix, player, passing=False):
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
                self.addEntryToDic(prevBoardVector, prevBoardVector, currBoardVector, passing)
            else:
                invPrevBoardVector = np.zeros(9 * 9, dtype=np.int32)
                for count in range(len(prevBoardVector)):
                    if prevBoardVector[count] != 0:
                        invPrevBoardVector[count] = -1 * prevBoardVector[count]
                    else:
                        invPrevBoardVector[count] = 0
                self.addEntryToDic(invPrevBoardVector, prevBoardVector, currBoardVector, passing)

    def addEntryToDic(self, entryBoardVector, prevBoardVector, currBoardVector, passing):
        move = np.absolute(currBoardVector - prevBoardVector)
        if self.dbFlagMoves == True:
            con = sqlite3.connect(r"DB's/MoveDB's/" + self.dbNameMoves, detect_types=sqlite3.PARSE_DECLTYPES)
            cur = con.cursor()
            if passing == False:
                cur.execute("insert into test values (?, ?, ?)",
                            (None, entryBoardVector, np.append(np.absolute(currBoardVector - prevBoardVector), 0)))
            else:
                cur.execute("insert into test values (?, ?, ?)", (None, entryBoardVector, np.copy(self.passVector)))
            con.commit()
        if self.dbFlagDist == True:
            con = sqlite3.connect(r"DB's/DistributionDB's/" + self.dbNameDist, detect_types=sqlite3.PARSE_DECLTYPES)
            cur = con.cursor()
            if passing == False:
                cur.execute("select count(*) from test where board = ?", (prevBoardVector,))
                data = cur.fetchall()
                if data[0][0] == 0:
                    cur.execute("insert into test values (?, ?, ?)",
                                (None, entryBoardVector, np.append(np.absolute(currBoardVector - prevBoardVector), 0)))
                else:
                    cur.execute("select distribution from test where board = ?", (prevBoardVector,))
                    old_dist = cur.fetchall()
                    cur.execute("UPDATE test SET distribution = ? WHERE board = ?",  (old_dist[0][0] + np.append(np.absolute(currBoardVector - prevBoardVector), 0), entryBoardVector))
            else:
                cur.execute("select count(*) from test where board = ?", (prevBoardVector,))
                data = cur.fetchall()
                if data[0][0] == 0:
                    cur.execute("insert into test values (?, ?, ?)",
                                (None, entryBoardVector, np.copy(self.passVector)))
                else:
                    cur.execute("select distribution from test where board = ?", (prevBoardVector,))
                    old_dist = cur.fetchall()
                    cur.execute("UPDATE test SET distribution = ? WHERE board = ?",
                                (old_dist[0][0] + np.copy(self.passVector), prevBoardVector))
            con.commit()

        else:
            if passing==False:
                if Hashable(entryBoardVector) in self.dic:
                    self.dic[Hashable(entryBoardVector)] += np.append(np.absolute(currBoardVector - prevBoardVector), 0)
                else:
                    self.dic[Hashable(entryBoardVector)] = np.append(np.absolute(currBoardVector - prevBoardVector), 0)
            else:
                if Hashable(entryBoardVector) in self.dic:
                    self.dic[Hashable(entryBoardVector)] += np.copy(self.passVector)

                else:
                    self.dic[Hashable(entryBoardVector)] = np.copy(self.passVector)

# end class TrainingDataSgfPass
def dbTest2():
    dbName = 'dan_data_10'
    con = sqlite3.connect(r"DB's/DistributionDB's/" + dbName, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("Select * from test where id = 1")
    data = cur.fetchall()
    print(data)
    con.close()
dbTest2()

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
    #ranged=1000
    #t = TrainingDataSgf("dgs",range(0,ranged))
    t=TrainingDataSgf("dgs","dan_data_10")
    end = time.clock()
    length = 0
    for entry in t.dic:
        length += 1
    l.append(length)
    #print("Range:",ranged)
    print("imported distinct boards:",length)
    print("importing took", np.round(end - start,3), "seconds")
#test2()

def test3():
    start = time.clock()
    t = TrainingDataSgf("dgs", range(10000))

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
#test3()

def test3pass():
    start = time.clock()
    t = TrainingDataSgfPass("dgs")

    zeroMatrix = t.dic[Hashable(np.zeros(t.n * t.n, dtype=np.int32))]
    print('\n', zeroMatrix[:-1].reshape((9, 9)), '\n')
    print("passing: ", zeroMatrix[81])
    print("entries: ", np.sum(zeroMatrix))

    # print cumulated move distributions for boards with exactly one stone
    secondMoveDist = np.zeros(9 * 9+1, dtype=np.int32)
    for entry in t.dic:
        thisis = Hashable.unwrap(entry)
        if np.sum(np.absolute(thisis)) == 1:
            secondMoveDist += t.dic[entry]
    print(secondMoveDist[:-1].reshape((9, 9)))
    print("passing: ", secondMoveDist[81])
    print("entries: ",np.sum(secondMoveDist))
    print("\nTime " + str(time.clock() - start))
#test3pass()

def dbCreate():
    TrainingDataSgfPass(folder="dgs", id_list = 'dan_data_10', dbNameMoves="dan_data_10")
    con = sqlite3.connect(r"DB's/MoveDB's/dan_data_10", detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("select * from test where id <= 100")
    data = cur.fetchall()
    con.close
    print(data)
#dbCreate()

def distDbCreate():
    TrainingDataSgfPass(folder="dgs", id_list='dan_data_10', dbNameDist="dan_data_10")
    con = sqlite3.connect(r"DB's/DistributionDB's/dan_data_10", detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("select * from test where id <= 100")
    data = cur.fetchall()
    con.close()
    print(data)
#distDbCreate()

