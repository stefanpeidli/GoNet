# -*- coding: utf-8 -*-
from Board import *
from PolicyNet import *
import time

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
    def __init__(self, n):
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
    def __init__(self,n,weights_file="ambtestfilt.npz"):
        super(IntelligentEngine, self).__init__(n)
        self.PolicyNet = PolicyNet([9*9,1000,200,9*9+1]) #untrained
        self.PolicyNet.loadweightsfromfile(weights_file)

    def version(self):
        return "2.0"

    # need to get move from Neural Network here (forward propagate the current board)
    def play_legal_move(self, board, stone):
        if(stone == Stone.Black):
            out=self.PolicyNet.propagate_board(board)
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
            out=self.PolicyNet.propagate_board(board)
            board.vertices = tempVertices
        print("Network Output:")
        print(np.round(out[:-1].reshape((9,9)),2))
        while sum(out) > 0:
            move=np.argmax(out)
            #print(move)
            if move == 81: #passing is always legal. 82er eintrag (?)
                print("The Policy Network considers passing as the best move with a relative confidence of",str(round(out[move]*100))+"%",".")
                return "pass"
            x=int(move%9)
            y=int(np.floor(move/9))
            coords=(x,y) #check if this is right, i dont think so. The counting is wrong
            if board.play_is_legal(coords[0],coords[1], stone):
                board.play_stone(coords[0],coords[1], stone)
                print("The Policy Network considers",coords,
                      "as the best move with a relative confidence of",str(round(out[move]*100))+"%",".")
                return coords
            else:
                out[move]=0
        print("The Policy Network considers passing as the best move with a relative confidence of THIS IS NO OUTPUT YET.")
        return "pass"


class PolicyEngine(BaseEngine):
    def __init__(self, n, weights_file='none', filter_ids=[0,1,2,3,4,5,6,7]):
        super(PolicyEngine, self).__init__(n)
        self.PolicyNet = PolicyNet([9*9,1000,200,9*9+1], filter_ids=filter_ids)  # untrained
        if weights_file is not "none":
            self.PolicyNet.loadweightsfromfile(weights_file)

    def version(self):
        return "2.0"

    # need to get move from Neural Network here (forward propagate the current board)
    def play_legal_move(self, board, stone, details=False):
        if stone == Stone.Black:
            out = self.PolicyNet.propagate_board(board)
        else:
            boardVector = board.vertices.flatten()
            invBoard = np.zeros(9 * 9, dtype=np.int32)
            for count in range(len(boardVector)):
                if boardVector[count] != 0:
                    invBoard[count] = -1 * boardVector[count]
                else:
                    invBoard[count] = 0
            tempVertices = np.copy(board.vertices)
            board.vertices = invBoard.reshape((9, 9))
            out = self.PolicyNet.propagate_board(board)
            board.vertices = tempVertices

        # We need to prohibit passing sometimes:
        our_stones = np.sum(board.vertices[board.vertices == stone])
        their_stones = np.sum(board.vertices[board.vertices == -stone])
        if our_stones + their_stones > 81 * 0.75 and our_stones / their_stones > 1.1:
            allow_passing = True
        else:
            allow_passing = False

        if details:
            print("Network Output:")
            print(np.round(out[:-1].reshape((9, 9)), 2))
        while sum(out) > 0:
            move = np.argmax(out)
            if move == 81 and allow_passing:  # passing is always legal. 82er eintrag (?)
                if details:
                    print("The Policy Network considers passing as the best move with a relative confidence of",
                          str(round(out[move]*100))+"%", ".")
                return "pass"
            elif move == 81 and not allow_passing:  # Get second best suggestion
                out[move] = 0
                move = np.argmax(out)
            x = int(move % 9)
            y = int(np.floor(move / 9))
            coords = (y, x)
            if board.play_is_legal(coords[0], coords[1], stone):
                board.play_stone(coords[0], coords[1], stone)
                if details:
                    print("The Policy Network considers", coords,
                          "as the best move with a relative confidence of", str(round(out[move]*100))+"%", ".")
                return coords
            else:
                out[move] = 0
        if details:
            print("The Policy Network considers passing as the best move with a relative confidence "
                  "of THIS IS NO OUTPUT YET.")
        return "pass"

    def play_legal_rand_move(self, board, stone, details=False):
        try_flag = 0
        while try_flag < 100:
            move = np.round(np.random.uniform(0, 81), 0)  # generate randome move
            if move == 81:  # passing is always legal. 82er eintrag (?)
                if details:
                    print("The random bot passes.")
                return "pass"
            x = int(move % 9)
            y = int(np.floor(move / 9))
            coords = (x, y)
            if board.play_is_legal(coords[0], coords[1], stone):
                board.play_stone(coords[0], coords[1], stone)
                if details:
                    print("The random bot plays at ", coords, ".")
                return coords
            try_flag += 1
        if details:
            print("Since only passing seems legal, the random bot passes.")
        return "pass"

    def play_against_random(self, speed, maxturns, details=False):
        engine = self
        if details:
            print("Game starts")
            engine.board.show()
        Players = [Stone.Black, Stone.White]
        PlayerNames = ["Black (PolicyBot)", "White (RandomBot)"]
        turn = 0
        temp = Board(9)
        # save = temp.vertices
        while turn < maxturns:
            if details:
                print("Now it is turn number", turn, "and", PlayerNames[np.mod(turn, 2)], "is playing.")
            if PlayerNames[np.mod(turn, 2)] == "Black (PolicyBot)":
                engine.play_legal_move(engine.board, Players[np.mod(turn, 2)], details)
            else:  # Random Bot
                engine.play_legal_rand_move(engine.board, Players[np.mod(turn, 2)], details)
            if details:
                print("After move was played board is:")
                engine.board.show()
            if turn > 3:
                if (engine.board.history[-1] == engine.board.vertices).all() and (engine.board.history[-1] == engine.board.history[-2]).all():
                    if details:
                        print("Game over by resign after", turn, "turns.")
                    return engine.board.history # TODO
            turn += 1
            time.sleep(speed)
        return engine.board.history
        
    def play_against_self(self, speed, maxturns, details=False):
        engine = self
        if details:
            print("Game starts")
            engine.board.show()
        Players = [Stone.Black, Stone.White]
        PlayerNames = ["Black", "White"]
        turn = 0
        flag = 0
        temp = Board(9)
        save = temp.vertices
        while turn < maxturns:
            if details:
                print("Now it is turn number", turn, "and", PlayerNames[np.mod(turn, 2)], "is playing.")
            engine.play_legal_move(engine.board, Players[np.mod(turn, 2)], details)
            if details:
                print("After move was played board is:")
                engine.board.show()
            if (save == engine.board.vertices).all():
                flag += 1
            else:
                flag = 0
            if flag == 2:
                if details:
                    print("Game over by resign after", turn, "turns.")
                return engine.board.history
            save = engine.board.vertices
            turn += 1
            time.sleep(speed)
        return engine.board.history


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
    engine.board.play_stone(1, 0, Stone.Black)
    engine.board.play_stone(0, 1, Stone.Black)
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
    engine.board.play_stone(1, 0, Stone.Black)
    engine.board.play_stone(0, 1, Stone.Black)
    engine.board.show()
    engine.play_legal_move(engine.board, Stone.White)
    engine.board.show()

#test3()
    
def test4():
    engine = PolicyEngine(9)
    engine.board.play_stone(1, 0, Stone.Black)
    engine.board.play_stone(0, 1, Stone.Black)
    engine.board.show()
    engine.play_legal_move(engine.board, Stone.White)
    engine.board.show()

#test4()
    
def test5():
    engine = PolicyEngine(9)
    game_history = engine.play_against_self(0, 100, details=False)
    final_board = game_history[-1]
    print("protoscore: Black", np.sum(final_board[final_board == Stone.Black])/Stone.Black)
    print("protoscore: White", np.sum(final_board[final_board == Stone.White])/Stone.White)

    
#test5()

def test6():
    engine = PolicyEngine(9, "ambitestfilt1234567logrule")
    game_history = engine.play_against_random(0, 100, details=False)
    final_board = game_history[-1]
    print("protoscore: Black", np.sum(final_board[final_board == Stone.Black]) / Stone.Black)
    print("protoscore: White", np.sum(final_board[final_board == Stone.White]) / Stone.White)

#test6()

def test7():
    engine = PolicyEngine(9, "ambitestfilt1234567logrule", filter_ids=[0,1,2,3,4,5,6,7])
    winner = []
    rounds = 100
    komi = 6

    for i in range(rounds):
        game_history = engine.play_against_random(0, 100, details=False)
        final_board = game_history[-1]
        b = np.sum(final_board[final_board == Stone.Black]) / Stone.Black
        w = np.sum(final_board[final_board == Stone.White]) / Stone.White
        w += komi
        # PROTOSCORE
        if b > w:
            winner.append("B")
        elif b == w:
            winner.append("D")
        else:
            winner.append("W")
        print("Game", i, ":", b, w)

    # print(winner)
    print()
    print("The komi was: ", komi)
    print("The PolicyNet won", winner.count("B"), "out of", rounds, "games against the Random Bot.")
    print(winner.count("D"), "games ended in a draw.")

#test7()
