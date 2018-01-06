# -*- coding: utf-8 -*-
from Board import *
from PolicyNetForExecutable import *
from FilterNet import *
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
        self.PolicyNet.loadweightsfromfile("Saved_Weights"
                        ,'weights1801011848eta100001000epochs1000batchsize1.npz')

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
        print(np.round(out[:-1].reshape((9,9)),2))
        while sum(out) > 0:
            move=np.argmax(out)
            print(move)
            if move == 81: #passing is always legal. 82er eintrag (?)
                print("The Policy Network considers passing as the best move with a relative confidence of",str(round(out[move]*100))+"%",".")
                return "pass"
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
        
class FilterEngine(BaseEngine):
    def __init__(self,n):
        super(FilterEngine, self).__init__(n)
        self.FilterNet = FilterNet([9*9,1000,200,9*9+1]) #untrained
        self.FilterNet.loadweightsfromfile("ambtestfilt")

    def version(self):
        return "2.0"

    # need to get move from Neural Network here (forward propagate the current board)
    def play_legal_move(self, board, stone):
        if(stone == Stone.Black):
            out=self.FilterNet.Propagate(board)
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
            out=self.FilterNet.Propagate(board)
            board.vertices = tempVertices
        print("Network Output:")
        print(np.round(out[:-1].reshape((9,9)),2))
        while sum(out) > 0:
            move=np.argmax(out)
            #print(move)
            if move == 81: #passing is always legal. 82er eintrag (?)
                print("The Filter Network considers passing as the best move with a relative confidence of",str(round(out[move]*100))+"%",".")
                return "pass"
            x=int(move%9)
            y=int(np.floor(move/9))
            coords=(x,y) #check if this is right, i dont think so. The counting is wrong
            if board.play_is_legal(coords[0],coords[1], stone):
                board.play_stone(coords[0],coords[1], stone)
                print("The Filter Network considers",coords,
                      "as the best move with a relative confidence of",str(round(out[move]*100))+"%",".")
                return coords
            else:
                out[move]=0
        print("The Filter Network considers passing as the best move with a relative confidence of THIS IS NO OUTPUT YET.")
        return "pass"  
        
    def play_against_self(self,speed,maxturns):
        engine = FilterEngine(9)
        print("Game starts")
        engine.board.show()
        Players=[Stone.Black,Stone.White]
        PlayerNames=["Black","White"]
        turn=0
        flag=0
        temp = Board(9)
        save= temp.vertices
        while turn <maxturns:
            print("Now it is turn number",turn,"and",PlayerNames[np.mod(turn,2)],"is playing.")
            engine.play_legal_move(engine.board, Players[np.mod(turn,2)])
            print("After move was played board is:")
            engine.board.show()
            if (save==engine.board.vertices).all():
                flag+=1
            else:
                flag=0
            if flag == 2:
                print("Game over by resign after",turn,"turns.")
                return
            save = engine.board.vertices
            turn +=1
            time.sleep(speed)
        #TODO: netz spielt gespiegelt? output/Vorhersage stimmt nicht mit zug überein (ist nicht legal arg max)

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
    
def test4():
    engine = FilterEngine(9)
    engine.board.play_stone(1,0,Stone.Black)
    engine.board.play_stone(0,1,Stone.Black)
    engine.board.show()
    engine.play_legal_move(engine.board, Stone.White)
    engine.board.show()

#test4()
    
def test5():
    engine = FilterEngine(9)
    engine.play_against_self(0.1,100)
    
test5()