# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:34:08 2017

@author: paddy
"""
import numpy as np

class Stone:
    Empty = 0
    Black = -1
    White = 1

colour_names = { Stone.Empty:"Empty", Stone.Black:"Black", Stone.White:"White", }

def flipped_stone(stone):
    if stone == Stone.Empty:
        return Stone.Empty
    elif stone == Stone.Black:
        return Stone.White
    else:
        return Stone.Black

# Standardbasis des R2 (Das ist keine Basis du Lurch :P)
dxdys = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
class Board:
    def __init__(self, N):
        self.N = N
        self.clear()

    def clear(self):
        self.vertices = np.empty((self.N, self.N), dtype=np.int32)
        self.vertices.fill(Stone.Empty)
        self.history = []
        
    def is_on_board(self, x, y):
        return 0 <= x < self.N and 0 <= y < self.N

    # liefert true, wenn die Gruppe, in der sich (start_x, start_y) befindet mindestens eine Freiheit besitzt
    # Wenn capture = true im Input und Gruppe hat keine Freiheiten: Gruppe wird vom Board entfernt

    def check_group_liberties(self, start_x, start_y, capture=False):
        group = [(start_x, start_y)]
        i = 0
        while i < len(group):
            x, y = group[i]
            i += 1
            for dx, dy in dxdys:
                adj_x, adj_y = x + dx, y + dy
                if self.is_on_board(adj_x, adj_y) and not (adj_x, adj_y) in group:
                    if self.vertices[adj_x, adj_y] == Stone.Empty:
                        return True
                    elif self.vertices[adj_x, adj_y] == self.vertices[start_x, start_y]:
                        group.append((adj_x, adj_y))
                        
        if capture:
            for x, y in group:
                self.vertices[x, y] = Stone.Empty
                
        return False
    

    # Stein wird gespielt. Dann wird geschaut ob andere Gruppe dadurch gefangen wird. Dann wird Ko gecheckt
    # Benutzt check_group_liberties(..) um zu checken ob der Zug legal ist und spielt den Stein falls ja
    def play_stone(self, x, y, stone, just_testing=False):
        if self.vertices[x, y] != Stone.Empty:
            return False
        self.vertices[x, y] = stone
        made_capture = False

        for dx, dy in dxdys:
            adj_x, adj_y = x+dx, y+dy
            if self.is_on_board(adj_x, adj_y) and self.vertices[adj_x, adj_y] == flipped_stone(self.vertices[x, y]):
                if not self.check_group_liberties(adj_x, adj_y, capture=True):
                    made_capture = True
        # superko
        for prev_vertices in self.history:
            if np.array_equal(prev_vertices, self.vertices):
                return False

        if not just_testing:
            self.history.append(np.copy(self.vertices))

        return made_capture or self.check_group_liberties(x, y)
                
    def play_is_legal(self, x, y, stone):
        saved_vertices = np.copy(self.vertices)
        move_is_legal = self.play_stone(x, y, stone, just_testing=True)
        self.vertices = saved_vertices
        return move_is_legal
    
    def flip_colors(self):
        for x in range(self.N):
            for y in range(self.N):
                self.vertices[x, y] = flipped_stone(self.vertices[x, y])
                
    def show(self):
        stone_strings = {
                Stone.Empty: '.',
                Stone.Black: '\033[31m0\033[0m',
                Stone.White: '\033[37m0\033[0m' }
        Board_repres=np.empty((self.N,self.N),str)
        for x in range(self.N):
            print("=")
        print("")
        for y in range(self.N):
            for x in range(self.N):
                Board_repres[x,y] = stone_strings[self.vertices[x,y]]
                print(stone_strings[self.vertices[x,y]])
            print("")
        for x in range(self.N):
            print("=")
        print("")
        print(Board_repres)

    
def show_sequence(board, moves, first_color):
    board.clear()
    color = first_color
    for x, y in moves:
        legal = board.play_stone(x, y, color)
        board.show()
        print("move was legal?", legal)
        color = flipped_stone(color)


def test_Board():
    board = Board(5)

    print("simplest capture:")
    show_sequence(board, [(1, 0), (0, 0), (0, 1)], Stone.Black)
    print("move at (0, 0) is legal?", board.play_is_legal(0, 0, Stone.White))
    board.flip_colors()

    print("bigger capture:")
    show_sequence(board, [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4)], Stone.Black)

    print("ko:")
    show_sequence(board, [(0, 1), (3, 1), (1, 0), (2, 0), (1, 2), (2, 2), (2, 1), (1, 1)], Stone.Black)
    print("move at (2, 1) is legal?", board.play_is_legal(2, 1, Stone.Black))
    board.show()
    board.flip_colors()
    print("fipped board:")
    board.show()

    print("self capture:")
    show_sequence(board, [(0, 1), (1, 1), (1, 0)], Stone.Black)
    print("move at (0, 0) is legal?", board.play_is_legal(0, 0, Stone.White))


#test_Board()