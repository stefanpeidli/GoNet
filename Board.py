# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:34:08 2017

@author: paddy
"""
import numpy as np


class Stone:
    Empty = 0
    Black = 1
    White = 2
    
dxdys = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
class Board:
    def __init__(self, N):
        self.N = N
        self.clear()

    def clear(self):
        self.vertices = np.empty((self.N, self.N), dtype = np.int32)
        self.vertices.fill(Stone.Empty)
        self.history = []
        
    def is_on_board(self, x, y):
        return x >= 0 and x < self.N and y >= 0 and y < self.N
    
    def check_group_liberties(self, start_x, start_y, capture=False):
        self.group = [(start_x, start_y)]
        
        i = 0
        while i < len(self.group):
            x,y = self.group[i]
            i+=1
            for dx,dy in dxdys:
                adj_x, adj_y = x + dx, y + dy
                if self.is_on_board(adj_x,adj_y) and not self.group.__contains__((adj_x, adj_y)):
                    if self.vertices[adj_x, adj_y] == Stone.Empty:
                        return True
                    elif self.vertices[adj_x, adj_y] == self.vertices[start_x, start_y]:
                        self.group.append((adj_x, adj_y))
       
                    
        if capture:
            for x,y in self.group:
                self.vertices[x,y] = Stone.Empty
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
        