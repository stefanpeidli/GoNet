# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 20:54:24 2018

@author: Stefan
"""
import numpy as np
from Hashable import Hashable
import os
import sys
import sqlite3
from Filters import filter_eyes
from Filters import filter_captures



def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

def relu(x):
    x[x < 0] = 0
    return x

class PolicyNet:
    def __init__(self, layers=[9*9, 1000, 100, 9*9+1], activation_function=0):
        # Specifications of the game
        self.n = 9  # 9x9 board
        self.filtercount = 2
        layers[0] += self.filtercount*self.n*self.n

        # Parameters of the NN
        self.layers = layers  # please leave the first and last equal zu n^2 for now
        self.activation_function = activation_function

        # Initialize the weights
        self.layercount = len(self.layers)-1
        self.init_weights()

        if self.activation_function is 0:  #TODO wie im last layer haben wir softmax! wie initialisieren???
            mu = 0
            self.weights = [0]*self.layercount  # alloc memory
            for i in range(0, self.layercount):
                sigma = 1/np.sqrt(self.layers[i+1])
                self.weights[i] = np.random.normal(mu, sigma, (self.layers[i+1], self.layers[i]+1))  # the +1 in the input dimension is for the bias

        elif self.activation_function is 1:
            mu = 0
            self.layercount = len(self.layers)-1
            self.weights = [0]*self.layercount  # alloc memory
            for i in range(0, self.layercount):
                sigma = np.sqrt(2)/np.sqrt(self.layers[i+1])
                self.weights[i] = np.random.normal(mu, sigma, (self.layers[i+1], self.layers[i]+1))  # the +1 in the input dimension is for the bias

    def apply_filters(self, board, color = -1):
        filtered_1 = filter_eyes(board,color)
        filtered_2 = filter_captures(board,color)
        return [filtered_1.flatten(),filtered_2.flatten()]

    # Function Definition yard

    # error functions

    # error fct Number 0
    def compute_KL_divergence(self, suggested, target):  # Compute Kullback-Leibler divergence, stabilized version
        t = target[target != 0]  # ->we'd divide by 0 else, does not have inpact on error anyway
        s = suggested[target != 0]
        difference = s / t  # this is stable
        error = - np.inner(t*np.log(difference), np.ones(len(t)))
        return error

    # error fct Number 1
    def compute_ms_error(self, suggested, target):  # Returns the total mean square error
        difference = np.absolute(suggested - target)
        error = 0.5 * np.inner(difference, difference)
        return error

    # error fct Number 2
    def compute_Hellinger_dist(self, suggested, target):
        return np.linalg.norm(np.sqrt(suggested) - np.sqrt(target), ord=2) / np.sqrt(2)

    # error fct Number 3
    def compute_cross_entropy(self, suggested, target):
        return self.compute_entropy(target) + self.compute_KL_divergence(target, suggested)

    # error fct Number x, actually not a good one. Only for statistics.
    def compute_abs_error(self, suggested, target):  # compare the prediction with the answer/target, absolute error
        difference = np.absolute(suggested - target)
        error = np.inner(difference, np.ones(len(target)))
        return error

    def compute_entropy(self, distribution):
        return -np.inner(distribution, np.log(distribution))

    # Auxilary and Utilary Functions

    def init_weights(self):
        # Xavier Initialization
        if self.activation_function is 0:  # TODO wie im last layer haben wir softmax! wie initialisieren???
            mu = 0
            self.weights = [0] * self.layercount  # alloc memory
            for i in range(0, self.layercount):
                sigma = 1 / np.sqrt(self.layers[i + 1])
                self.weights[i] = np.random.normal(mu, sigma, (self.layers[i + 1], self.layers[i] + 1))
        # He Initialization
        elif self.activation_function is 1:
            mu = 0
            self.weights = [0] * self.layercount  # alloc memory
            for i in range(0, self.layercount):
                sigma = np.sqrt(2) / np.sqrt(
                    self.layers[i + 1])
                self.weights[i] = np.random.normal(mu, sigma, (
                    self.layers[i + 1], self.layers[i] + 1))  # edit: the +1 in the input dimension is for the bias

    def weight_ensemble(self, testset, instances=1, details=False):
        optimal_weights = self.weights
        optimal_value = self.PropagateSet(testset)
        first_value = optimal_value
        for i in range(instances):
            self.init_weights()
            weights = self.weights
            value = self.PropagateSet(testset)
            if value < optimal_value:
                optimal_value = value
                optimal_weights = weights
        improvement = first_value - optimal_value
        self.weights = optimal_weights
        if details:
            return improvement

    def convert_input(self, boardvector):  # rescaling help function [-1,0,1]->[-1.35,0.45,1.05]
        boardvector = boardvector.astype(float)
        for i in range(0, len(boardvector)):
            if boardvector[i] == 0:
                boardvector[i] = 0.45
            if boardvector[i] == -1:
                boardvector[i] = -1.35
            if boardvector[i] == 1:
                boardvector[i] = 1.05
        return boardvector



    def extract_batch_from_db(self, db_name, batchsize, enrichment=False):
        con = sqlite3.connect(r"DB/Dist/" + db_name, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = con.cursor()
        #cur.execute("select count(*) from test")
        #data = cur.fetchall()
        #datasize = data[0][0]  # number of boards present in the database
        #number_of_batchs = int(np.ceil(datasize / batchsize))
        if not enrichment:
            cur.execute("select * from test order by Random() Limit " + str(batchsize))
        else:
            cur.execute("select * from dist order by Random() Limit " + str(batchsize))
        batch = cur.fetchall()
        con.close()
        return batch

    def saveweights(self, filename, folder='Saved_Weights'):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = dir_path + "/" + folder + "/" + filename
        np.savez(file, self.weights)



    def loadweightsfromfile(self, filename, folder='Saved_Weights'):
        # if file doesnt exist, do nothing
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_path, folder+"/"+filename)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = dir_path + "/" + folder + "/" + filename
        if os.path.exists(path):
            with np.load(path) as data:
                self.weights = []
                self.layer = [data['arr_0'][0].shape[1]]  # there are n+1 layers if there are n weight matrices
                for i in range(len(data['arr_0'])):
                    self.weights.append(data['arr_0'][i])
                    tempshape = data['arr_0'][i].shape
                    self.layer.append(tempshape[0])
                self.layercount = len(self.layer) - 1
        elif os.path.exists(path + ".npz"):
            with np.load(path + ".npz") as data:
                self.weights = []
                for i in range(len(data['arr_0'])):
                    self.weights.append(data['arr_0'][i])



    def Propagate(self, board):
        # Convert board to NeuroNet format (82-dim vector)
        if type(board) != list and type(board) != np.ndarray:
            board = board.vertices
        if len(board) != 82:
            board = board.flatten()
        board = np.asarray(board, float)
        # Like Heining we are setting: (-1.35:w, 0.45:empty, 1.05:b)
        [t1, t2] = self.apply_filters(board.reshape((9, 9)))
        for i in range(0, len(board)):
            if board[i] == np.int(0):
                board[i] = 0.45
            if board[i] == -1:
                board[i] = -1.35
            if board[i] == 1:
                board[i] = 1.05
        board = [*board, *t1, *t2]
        y = np.append(board, [1])
        # Forward-propagate
        for i in range(0, self.layercount):
            W = self.weights[i]
            s = W.dot(y)
            if i == self.layercount-1:  # softmax as activationfct only in last layer
                y = np.append(softmax(s), [1])
            elif self.activation_function is 0:
                y = np.append(np.tanh(s), [1])
            elif self.activation_function is 1:
                y = np.append(relu(s), [1])
        out = y[:-1]
        return out

    def PropagateSet(self, testset, db=False, adaptive_rule='linear', error_function=0):
        error = 0
        checked = 0
        if not db:
            given_set = testset.dic
        else:
            pass  # TODO
        for entry in given_set:
            if not db:
                t0 = Hashable.unwrap(entry)
                [t1, t2] = self.apply_filters(t0.reshape((9, 9)))
                testdata = [*self.convert_input(t0), *t1, *t2]
                targ = testset.dic[entry].reshape(9*9+1)
            else:
                pass  # TODO
            if np.sum(targ) > 0:  # We can only learn if there are actual target vectors
                targ_sum = np.sum(targ)
                targ = targ/np.linalg.norm(targ, ord=1)  # normalize (L1-norm)
                y = np.append(testdata, [1])
                # Forward-propagate
                for i in range(0, self.layercount):
                    W = self.weights[i]
                    s = W.dot(y)
                    if i == self.layercount - 1:  # softmax as activationfct only in last layer
                        y = np.append(softmax(s), [1])
                    elif self.activation_function is 0:
                        y = np.append(np.tanh(s), [1])
                    elif self.activation_function is 1:
                        y = np.append(relu(s), [1])
                # sum up the error
                if error_function is 0:
                    error += self.compute_KL_divergence(y[:-1], targ)
                elif error_function is 1:
                    error += self.compute_ms_error(y[:-1], targ)
                elif error_function is 2:
                    error += self.compute_Hellinger_dist(y[:-1], targ)

                if adaptive_rule == "linear":
                    checked += 1 * targ_sum
                elif adaptive_rule == "logarithmic":
                    checked += 1 * np.log(2 + targ_sum)
                elif adaptive_rule == "none":
                    checked += 1
        if checked is 0:
            print("The Set contained no feasible boards to propagate, sorry")
            return
        else:
            error = error / checked  # average over the test set
            return error


