"""
@author: Stefan Peidli
License: MIT
Tags: Neural Network
"""

import numpy as np


n = 9

# Testboards
def gen_test_board(method=0):
    if method == 0:
        b = np.zeros((n, n))
        b[0, 2] = 1
        b[1, 3] = 1
        b[3, 3] = 1
        b[2, 3] = -1
        b[0, 1] = -1
        b[1, 0] = -1
        b[1, 1] = -1
        b[2, 2] = 1
    if method == 1:
        b = np.round(np.random.uniform(-1, 1, (n, n)), 0)
    return b

gen_test_board(1)

dxdys = [(1, 0), (0, 1), (-1, 0), (0, -1)]

# help functions


def is_on_board(n, x, y):
        return 0 <= x < n and 0 <= y < n


# TODO da ist irgendwo ein fehler. libs stimmen manchmal nicht
def give_group_at_position(board, start_x, start_y):
        group = [(start_x, start_y)]
        checked = []
        i = 0
        liberts = 0
        while i < len(group):
            x, y = group[i]
            i += 1
            for dx, dy in dxdys:
                adj_x, adj_y = x + dx, y + dy
                if is_on_board(board.shape[0], adj_x, adj_y) and not (adj_x, adj_y) in group:
                    if board[adj_x, adj_y] == 0 and not (adj_x, adj_y) in checked:
                        liberts += 1
                        checked.append((adj_x, adj_y))
                    elif board[adj_x, adj_y] == board[start_x, start_y]:
                        group.append((adj_x, adj_y))
        if board[start_x, start_y] == 0:
            liberts = 0
        return [group, liberts]



def give_liberties(board,color):
    libs = np.zeros((n, n))
    for row in range(n):
        for col in range(n):
            if board[row, col] == color:
                [g, li] = give_group_at_position(board, row, col)
                libs[row,col] = li
    return libs

# Filters
    
# Filters that are self-mappings


# Eyes . ID = 0
# shows the eyes of player color
def filter_eyes(board, color):
    n = board.shape[0]
    eyes = np.zeros((n, n))
    board = board * color
    for row in range(n):
        for col in range(n):
            if board[row, col] == 0:  # only free fields can be eyes
                if not(row == 0):
                    eyes[row, col] += board[row-1,col]
                if not(row == n-1):
                    eyes[row, col] += board[row+1,col]
                if not(col == 0):
                    eyes[row, col] += board[row,col-1]
                if not(col == n-1):
                    eyes[row, col] += board[row,col+1]
    eyes[0, :] += 1
    eyes[-1, :] += 1
    eyes[:, 0] += 1
    eyes[:, -1] += 1
    eyes[eyes != 4] = 0
    eyes = eyes / 4
    return eyes


# Shows which move will result in an eye being created (1) or destroyed (-1) . ID = 1.
# Note: Eyes by capture are created by capturing a single stone
def filter_eyes_create(board, color=1):
    board.reshape((9, 9))
    n = board.shape[0]
    reyes = np.zeros((n, n))
    eyc = np.sum(filter_eyes(board, color))  # current eyes
    cap = filter_captures(board, color)
    for row in range(n):
        for col in range(n):
            if board[row, col] == 0:  # only free fields can be set
                temp = board * 1  # python magic
                temp[row, col] = color
                eyn = np.sum(filter_eyes(temp, color))  # eyes by free creation
                # actually not good line below: we can also capture two single stones with one move..
                if cap[row, col] == 1:  # capture one eye
                    eyn += 1
                reyes[row, col] = eyn - eyc
    return reyes
                

# captures ID = 2
# Shows how many stones player "color" (1=b,-1=w) would capture by playing a 
# move on a field
def filter_captures(board, color):
    board.reshape((9, 9))
    n = board.shape[0]
    cap = np.zeros((n, n))
    for row in range(n):
        for col in range(n):
            if board[row, col] == 0:  # only free fields can be set
                val = 0
                if not(row == 0):
                    if color == board[row-1, col] * -1:  # then is enemy
                        [group, libs] = give_group_at_position(board, row-1, col)
                        if libs == 1:
                            val = max(val, len(group))
                if not(row == n-1):
                    if color == board[row+1, col] * -1:
                        [group, libs] = give_group_at_position(board, row+1, col)
                        if libs == 1:
                            val = max(val, len(group))
                if not(col == 0):
                    if color == board[row, col-1] * -1:
                        [group, libs] = give_group_at_position(board, row, col-1)
                        if libs == 1:
                            val = max(val, len(group))
                if not(col == n-1):
                    if color == board[row, col+1] * -1:
                        [group, libs] = give_group_at_position(board, row, col+1)
                        if libs == 1:
                            val = max(val, len(group))
                cap[row, col] = val
    return cap


# rewards connecting groups and adding liberties to groups. But e.g. punishes
# playing a move into an own eye. ID = 3.
def filter_add_liberties(board, color):
    board.reshape((9, 9))
    n = board.shape[0]
    libmat = np.zeros((n, n))
    for row in range(n):
        for col in range(n):
            val = 0
            if board[row, col] == 0:  # only free fields can be set
                temp = board * 1  # do not delete this
                temp[row, col] = color
                [g, li] = give_group_at_position(temp, row, col)
                checked = []
                neighbours = 0
                if not(row == 0):
                    if color == board[row-1, col]:
                        [group, libs] = give_group_at_position(board, row-1, col)
                        val += li - libs
                        neighbours += 1
                        checked.extend(group)
                if not(row == n-1):
                    if color == board[row+1, col]:
                        [group, libs] = give_group_at_position(board, row+1, col)
                        if group in checked:
                            libs = 0
                        else:
                            neighbours += 1
                            checked.extend(group)
                        val += li - libs
                if not(col == 0):
                    if color == board[row, col-1]:
                        [group, libs] = give_group_at_position(board, row, col-1)
                        if group in checked:
                            libs = 0
                        else:
                            neighbours += 1
                            checked.extend(group)
                        val += li - libs
                if not(col == n-1):
                    if color == board[row, col+1]:
                        [group, libs] = give_group_at_position(board, row, col+1)
                        if group in checked:
                            libs = 0
                        else:
                            neighbours += 1
                            checked.extend(group)
                        val += li - libs
            libmat[row, col] = val
    return libmat


# measures total liberties added if move is played. ID = 4
def filter_liberization(board, color):
    board.reshape((9, 9))
    n = board.shape[0]
    libmat = np.zeros((n, n))
    for row in range(n):
        for col in range(n):
            val = 0
            if board[row, col] == 0:  # only free fields can be set
                temp = board * 1  # do not delete
                temp[row, col] = color
                [g, li] = give_group_at_position(temp, row, col)
                val = li
                checked = []
                neighbours = 0
                if not(row == 0):
                    if color == board[row-1, col]:
                        [group, libs] = give_group_at_position(board, row-1, col)
                        val += - libs
                        neighbours += 1
                        checked.extend(group)
                if not(row == n-1):
                    if color == board[row+1, col]:
                        [group, libs] = give_group_at_position(board, row+1, col)
                        if group in checked:
                            libs = 0
                        else:
                            neighbours += 1
                            checked.extend(group)
                        val += - libs
                if not(col == 0):
                    if color == board[row, col-1]:
                        [group, libs] = give_group_at_position(board, row, col-1)
                        if group in checked:
                            libs = 0
                        else:
                            neighbours += 1
                            checked.extend(group)
                        val += - libs
                if not(col == n-1):
                    if color == board[row, col+1]:
                        [group, libs] = give_group_at_position(board, row, col+1)
                        if group in checked:
                            libs = 0
                        else:
                            neighbours += 1
                            checked.extend(group)
                        val += - libs
            libmat[row, col] = val
    return libmat


# Gives all groups with their sizes as field values at the member positions of
# a color. ID = 5.
def filter_groups(board, color):
    board.reshape((9, 9))
    n = board.shape[0]
    gps = np.zeros((n, n))
    for row in range(n):
        for col in range(n):
            if board[row, col] == color and gps[row, col] == 0:
                [g, li] = give_group_at_position(board, row, col)
                size = len(g)
                for member in g:
                    gps[member] = size
    return gps


# Gives all groups of size k of color. with_values=False unifies the output to 1
# no ID yet
def filter_groups_of_size_k(board, k, color, with_values=False):
    board.reshape((9, 9))
    n = board.shape[0]
    gps = np.zeros((n, n))
    for row in range(n):
        for col in range(n):
            if board[row, col] == color and gps[row, col] == 0:
                [g, li] = give_group_at_position(board, row, col)
                size = len(g)
                if size == k:
                    for member in g:
                        if with_values:
                            gps[member] = size
                        else:
                            gps[member] = 1
    return gps


# Gives all groups of color with exactly k UNSECURED eyes (i.e. only stones that
# form the eye are contained within the same group, not the diagonal stones)
# no ID yet
def filter_groups_eyes_unsec(board, k, color):
    board.reshape((9, 9))
    n = board.shape[0]
    res = np.zeros((n, n))
    eyes = filter_eyes(board, color)
    print(eyes)
    for row in range(n):
        for col in range(n):
            if eyes[row, col] == 1:
                temp = board * 1
                temp[row, col] = color
                [g, li] = give_group_at_position(temp, row, col)
                is_contained = True
                if not(row == 0) and (row-1, col) not in g:
                        is_contained = False
                if not(row == n-1) and (row+1, col) not in g:
                        is_contained = False
                if not(col == 0) and (row, col-1) not in g:
                        is_contained = False
                if not(col == n-1) and (row, col+1) not in g:
                        is_contained = False
                if is_contained:
                    for [x, y] in g:
                        res[x, y] += 1
                res[row, col] = 0
    res[res != k] = 0
    res[res == k] = 1
    return res


# Filters that are not self-mappings

# gives the board with only the stones of one color. ID = 6.
def filter_color_separation(board, color):
    temp = board * 1
    # Very interesting. by *1 we make sure temp is only a copy of
    # board and not board itself. Else this function changes the board!
    temp[temp != color] = 0
    return temp 


# The Summary Fiter Function
    
def apply_filters_by_id(board,filter_id):
    # TODO ?
    return True

    
# Tests
def test():
    b = gen_test_board(1)
    print("Board")
    print(b)
     
    white_eyes = filter_eyes(b,1)
    print("Eyes white")
    print(white_eyes)
    
    black_eyes = filter_eyes(b,-1)
    print("Eyes black")
    print(black_eyes)
    
    w_e_r = filter_eyes_create(b,1)
    print("Eyes white can create")
    print(w_e_r)
    
    b_e_r = filter_eyes_create(b,-1)
    print("Eyes black can create")
    print(b_e_r)
    
    libs_w = give_liberties(b,1)
    print("Liberties white")
    print(libs_w)
    
    libs_b = give_liberties(b,-1)
    print("Liberties black")
    print(libs_b)
        
    cap_w = filter_captures(b,1)
    print("Captures white")
    print(cap_w)
    
    cap_b = filter_captures(b,-1)
    print("Captures black")
    print(cap_b)
    
    add_lib_w = filter_add_liberties(b,1)
    print("Liberties added to groups of white")
    print(add_lib_w)
    
    add_lib_b = filter_add_liberties(b,-1)
    print("Liberties added to groups of black")
    print(add_lib_b)
    
    liber_w = filter_liberization(b,1)
    print("Liberization of white")
    print(liber_w)
    
    liber_b = filter_liberization(b,-1)
    print("Liberization of black")
    print(liber_b)
    
#test()