# -*- coding: utf-8 -*-
from Board import *
import random


class Engine:

    def __init__(self, N):
        self.create_board(N)

    def create_board(self, N):
        self.board = Board(N)

    def set_komi(self, komi):
        pass

    def player_passed(self, stone):
        pass

    def stone_played(self, x, y, stone):
        if self.board.play_is_legal(x, y, stone):
            self.board.play_stone(x, y, stone)

    def play_legal_move(self, board, stone):
        move_list = []
        for i in range(board.N):
            for j in range(board.N):
                if board.play_is_legal(i, j, stone):
                    move_list.append((i, j))
        if move_list == []:
            print("kein legaler zug möglich")
            return
        ch = random.choice(move_list)
        print(ch)
        board.play_stone(ch[0], ch[1], stone)

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