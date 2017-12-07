# -*- coding: utf-8 -*-
from Board import *
from PolicyNetForExecutable import *
import random


class BaseEngine(object):
    def __init__(self, n):
        self.create_board(n)

    def create_board(self, n):
        self.board = Board(n)

    def name(self):
        return "Akira"

    def version(self):
        assert False

    def set_komi(self, komi):
        pass

    def player_passed(self, stone):
        pass

    def stone_played(self, x, y, stone):
         if self.board.play_is_legal(x, y, stone):
             self.board.play_stone(x, y, stone)

    def play_legal_move(self, board, stone):
        assert False

class Engine(BaseEngine):
    def __init__(self,n):
        super(Engine, self).__init__(n)

    def version(self):
        return "1.0"

    def play_legal_move(self, board, stone):
        move_list = ["pass"]
        for i in range(board.N):
            for j in range(board.N):
                if board.play_is_legal(i, j, stone):
                    move_list.append((i, j))
        ch = random.choice(move_list)
        print(ch)
        if ch != "pass":
            board.play_stone(ch[0], ch[1], stone)
        return ch

class IntelligentEngine(BaseEngine):
    def __init__(self,n):
        super(IntelligentEngine, self).__init__(n)
        self.PolicyNet = PolicyNet() #untrained
        self.PolicyNet.loadweightsfromfile("Saved_Weights",'weights1712071559eta1000010epochs20batchsize1Dan10.npz')

    def version(self):
        return "2.0"

    # need to get move from Neural Network here (forward propagate the current board)
    def play_legal_move(self, board, stone):
        if(stone == Stone.Black):
            out=self.PolicyNet.Propagate(board)
        else:
            boardVector = board.vertices.flatten()
            invBoard = np.zeros(9 * 9, dtype=np.int32)
            for count in range(len(boardVector)):
                if boardVector[count] != 0:
                    invBoard[count] = -1 * boardVector[count]
                else:
                    invBoard[count] = 0
            tempVertices = np.copy(board.vertices)
            board.vertices = invBoard.reshape((9,9))
            out=self.PolicyNet.Propagate(board)
            board.vertices = tempVertices
        while sum(out) > 0:
            move=np.argmax(out) # Problem: What happens if this is not unique?
            x=int(move%9)
            y=int(np.floor(move/9))
            coords=(x,y) #check if this is right, i dont think so. The counting is wrong
            if board.play_is_legal(coords[0],coords[1], stone):
                board.play_stone(coords[0],coords[1], stone)
                print("The Policy Network considers",coords,
                      "as the best move with a distribution value of",str(round(out[move]*100))+"%",".")
                return coords
            else:
                out[move]=0
        print("The Policy Network considers passing as the best move with a relative confidence of THIS IS NO OUTPUT YET.")
        return "pass"
        
        

def test():
    engine = Engine(5)
    stone = Stone.Black
    for i in range(28):
        engine.play_legal_move(engine.board, stone)
        stone = flipped_stone(stone)
    engine.board.show()

#test()

def test2():
    engine = Engine(5)
    engine.board.play_stone(1,0,Stone.Black)
    engine.board.play_stone(0,1,Stone.Black)
    for i in range(22):
        engine.play_legal_move(engine.board, Stone.White)
    engine.board.show()
    engine.play_legal_move(engine.board, Stone.White)
    engine.board.show()
    engine.play_legal_move(engine.board, Stone.White)
    engine.board.show()
    engine.play_legal_move(engine.board, Stone.White)
    engine.board.show()

#test2()

def test3():
    engine = IntelligentEngine(9)
    #engine.PolicyNet=NN
    engine.board.play_stone(1,0,Stone.Black)
    engine.board.play_stone(0,1,Stone.Black)
    engine.board.show()
    engine.play_legal_move(engine.board, Stone.White)
    engine.board.show()

#test3()