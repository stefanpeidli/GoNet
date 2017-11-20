# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:38:11 2017

@author: Stefan
"""

import numpy as np

n=9

games=gameslist
""" Create test games
gamelength=2 #average 9x9 game length according to google is 45
numberofgames=6
games = np.random.uniform(-1.5,1.5,(numberofgames,gamelength,n*n)).round()
games[0][0]=np.zeros(n*n)
games[0][1]=np.zeros(n*n)
games[0][1][3]=1
games[0][2]=games[0][1]
games[0][2][2]=-1

games[1][0]=np.zeros(n*n)
games[1][1]=np.zeros(n*n)
games[1][1][5]=1
games[1][2]=games[1][1]
games[1][2][9]=-1
"""


def countstones(board): #counts the total number of stones on the boards
    return int(np.sum(abs(board)))

def findboard(nodes,board): #gives layer and index within layer of the board if found, else string
    order=countstones(board) #this gives the layer in which we will search for the board
    if order>=len(nodes):
        return "Board not found, the layer "+str(int(order))+" does not exist yet"
    else:
        layer=nodes[order]

    for boardindex in range(0,len(layer)):
        if int(np.sum(abs(board-layer[boardindex])))==0:
            return [int(order),int(boardindex)]
    return "Board not found"

def giveweights(nodes,edges,board): #gives weights (from which one can compute the distribution) with boards to a given board
    [order,index]=findboard(nodes,board) #missing: check if this is not error message
    return [edges[order][index],nodes[order+1]]

def augment(nodes,edges,game,weight): #augments the graph=nodes,edges by a game(currently all games same quality):
    for i in range(1,len(game)-1):#first and last (?) board not of interest
        preboard=game[i-1]
        pb=findboard(nodes,preboard)
        board=game[i]
        b=findboard(nodes,board) #check if that board already exists in the graph
        if type(b) is str: #board not yet in graph, add it and adjust edges accordingly
            order=countstones(board)
            
            #adds to board to layer of its order
            if order>=len(nodes):#generate an additional layer
                nodes.append([])
                edges.append([])
            else: #add edges from the new node to next layer
                edges[order].append([0]*len(nodes[order+1])) #we don't know yet where it is good to go from there
            nodes[order].append(board)
            print("board added")
            #add edges to all nodes from prelayer to the new node
            print("len of prelayer edges is",len(edges[pb[0]]))
            if len(edges[pb[0]])==0:
                   edges[pb[0]].append(weight) #what if we warp?
            else:
                for j in range(0,len(edges[pb[0]])):#BUG: this is always an empty range
                    print(j,"khkijldhjhaslkh")
                    if j==pb[1]:
                        edges[pb[0]][j].append(weight)
                    else:
                        edges[pb[0]][j].append(29)

        else: #board already in graph
            print(b,"board is known")
            print([pb[0]],[pb[1]],[b[1]])
            edges[pb[0]][pb[1]][b[1]]+=weight



#create a test starting graph
nodes=list(( [ [np.zeros(n*n)] , [games[0][1],games[1][1]],[games[0][2],games[1][2]] ] )) # root node and two test layers
edges=list(( [ [[1,20]] , [[3,-6],[10,4]] ] ))




print(findboard(nodes,games[1][1]))
print(giveweights(nodes,edges,games[1][1]))

#augment(nodes,edges,games[0],1)

    