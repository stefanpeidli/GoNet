"""
@author: Stefan Peidli
License: MIT
Tags: Neural Network
"""

import numpy as np


n = 4

b = np.zeros((n,n))
b[0,2]=1
b[1,3]=1
b[3,3]=1
b[2,3]=-1
b[0,1]=-1
b[1,0]=-1
b[1,1]=-1
#print(b)
#print()


dxdys = [(1, 0), (0, 1), (-1, 0), (0, -1)]

#help functions

def is_on_board(n, x, y):
        return 0 <= x < n and 0 <= y < n

def give_group_at_position(board, start_x, start_y):
        group = [(start_x, start_y)]
        i = 0
        liberts = 0
        while i < len(group):
            x, y = group[i]
            i += 1
            for dx, dy in dxdys:
                adj_x, adj_y = x + dx, y + dy
                if is_on_board(board.shape[0],adj_x, adj_y) and not (adj_x, adj_y) in group:
                    if board[adj_x, adj_y] == 0:
                        liberts += 1
                    elif board[adj_x, adj_y] == board[start_x, start_y]:
                        group.append((adj_x, adj_y))
        if board[start_x,start_y]==0:
            liberts=0
        return [group, liberts]

### Filters

# Eyes
# shows the eyes of player black (=-1)
def filter_eyes(board): #only gives black eyes or white? TODO
    n = board.shape[0]
    eyes = np.zeros((n,n))
    for row in range(n):
        for col in range(n):
            if board[row,col]==0:#only free fields can be eyes
                if not(row == 0):
                    eyes[row,col]+=board[row-1,col]
                if not(row == n-1):
                    eyes[row,col]+=board[row+1,col]
                if not(col == 0):
                    eyes[row,col]+=board[row,col-1]
                if not(col == n-1):
                    eyes[row,col]+=board[row,col+1]
    eyes[0,:]+=1
    eyes[-1,:]+=1
    eyes[:,0]+=1
    eyes[:,-1]+=1
    eyes[eyes!=4]=0
    eyes=eyes/4
    return eyes

# captures
# Shows how many stones player "color" (1=b,-1=w) would capture by playing a 
# move on a field
def filter_captures(board,color):
    board.reshape((9,9))
    n = board.shape[0]
    cap = np.zeros((n,n))
    for row in range(n):
        for col in range(n):
            if board[row,col]==0:#only free fields can be eyes
                val = 0
                if not(row == 0):
                    if color == board[row-1,col] * -1: #then is enemy
                        [group,libs] = give_group_at_position(board,row-1,col)
                        #print([row,col],[group,libs])
                        if libs==1:
                            val=max(val,len(group))
                if not(row == n-1):
                    if color == board[row+1,col] * -1:
                        [group,libs] = give_group_at_position(board,row+1,col)
                        if libs==1:
                            val=max(val,len(group))
                if not(col == 0):
                    if color == board[row,col-1] * -1:
                        [group,libs] = give_group_at_position(board,row,col-1)
                        if libs==1:
                            val=max(val,len(group))
                if not(col == n-1):
                    if color == board[row,col+1] * -1:
                        [group,libs] = give_group_at_position(board,row,col+1)
                        if libs==1:
                            val=max(val,len(group))
                cap[row,col] = val
    return cap


### Tests
"""
eyes = filter_eyes(b)
print("Eyes black",eyes)


kack=np.zeros((n,n))
for row in range(n):
    for col in range(n):
        [g,l]=give_group_at_position(b,row,col)
        kack[row,col]=l
        
print("Kack",kack)


cap_b = filter_captures(b,1)
print("Captures black",cap_b)
"""