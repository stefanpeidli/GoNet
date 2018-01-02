"""
Created on Wed Nov  8 11:00:05 2017
@author: Stefan Peidli
License: MIT
Tags: Policy-net, Neural Network
"""
import numpy as np
from Hashable import Hashable
import os


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def relu(x):
    x[x < 0] = 0
    return x


class PolicyNet:
    def __init__(self, layers=[9 * 9, 100, 200, 300, 200, 100, 9 * 9 + 1]):
        ### Specifications of the game
        self.n = 9  # 9x9 board

        ### Parameters of the NN
        self.eta = 0.001  # learning rate
        self.layers = layers  # please leave the first and last equal zu n^2 for now

        ### Initialize the weights
        # here by a normal distribution N(mu,sigma)
        self.layercount = len(self.layers) - 1
        mu = 0
        self.weights = [0] * self.layercount  # alloc memory
        for i in range(0, self.layercount):
            sigma = 1 / np.sqrt(
                self.layers[i + 1])  # vgl Heining #TODO:Check if inputs are well-behaved (approx. normalized)
            self.weights[i] = np.random.normal(mu, sigma, (
            self.layers[i + 1], self.layers[i] + 1))  # edit: the +1 in the input dimension is for the bias

        ### Alloc memory for the error statistics
        # Hint:b=a[:-1,:] this is a handy formulation to delete the last column, Thus ~W=W[:-1,1]

        # self.errorsbyepoch=[] #mean squared error
        # self.abserrorbyepoch=[] #absolute error
        # self.KLdbyepoch=[] #Kullback-Leibler divergence

    ### Function Definition yard

    # error functions

    # error fct Number 0
    def compute_KL_divergence(self, suggested, target):  # Compute Kullback-Leibler divergence, now stable!
        t = target[
            target != 0]  # ->we'd divide by 0 else, does not have inpact on error anyway ->Problem: We don't punish the NN for predicting non-zero values on zero target!
        s = suggested[target != 0]
        difference = s / t  # this is stable
        Error = - np.inner(t * np.log(difference), np.ones(len(t)))
        return Error

    # error fct Number 1
    def compute_ms_error(self, suggested, target):  # Returns the total mean square error
        difference = np.absolute(suggested - target)
        Error = 0.5 * np.inner(difference, difference)
        return Error

    # error fct Number 2
    def compute_Hellinger_dist(self, suggested, target):
        return np.linalg.norm(np.sqrt(suggested) - np.sqrt(target), ord=2) / np.sqrt(2)

    # error fct Number 3
    def compute_cross_entropy(self, suggested, target):
        return self.compute_entropy(target) + self.compute_KL_divergence(target, suggested)  # wie rum kldiv?

    # error fct Number x, actually not a good one. Only for statistics
    def compute_abs_error(self, suggested, target):  # compare the prediction with the answer/target, absolute error
        difference = np.absolute(suggested - target)
        Error = np.inner(difference, np.ones(len(target)))
        return Error

    def compute_entropy(self, distribution):
        return -np.inner(distribution, np.log(distribution))

    ### Auxilary and Utilary Functions

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

    def saveweights(self, filename, folder='Saved_Weights'):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = dir_path + "/" + folder + "/" + filename
        np.savez(file, self.weights)

    def loadweightsfromfile(self, filename, folder='Saved_Weights'):
        # if file doesnt exist, do nothing
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = dir_path + "/" + folder + "/" + filename
        if os.path.exists(file):
            with np.load(file) as data:
                self.weights = []
                self.layer = [data['arr_0'][0].shape[1]]  # there are n+1 layers if there are n weightmatrices
                for i in range(len(data['arr_0'])):
                    self.weights.append(data['arr_0'][i])
                    tempshape = data['arr_0'][i].shape
                    self.layer.append(tempshape[0])
                self.layercount = len(self.layer) - 1
        elif os.path.exists(file + ".npz"):
            with np.load(file + ".npz") as data:
                self.weights = []
                for i in range(len(data['arr_0'])):
                    self.weights.append(data['arr_0'][i])


    def Propagate(self, board, activation_function=0):
        # convert board to NeuroNet format (82-dim vector)
        board = board.vertices
        if len(board) != 82:
            board = board.flatten()
        # Like Heining we are setting: (-1.35:w, 0.45:empty, 1.05:b)
        for i in range(0, len(board)):
            if board[i] == 0:
                board[i] = 0.45
            if board[i] == -1:
                board[i] = -1.35
            if board[i] == 1:
                board[i] = 1.05

        y = np.append(board, [1])  # apply offset
        # Forward-propagate
        for i in range(0, self.layercount):
            W = self.weights[i]
            s = W.dot(y)
            if i == self.layercount - 1:  # softmax as activationfct only in last layer
                y = np.append(softmax(s), [1])  # We append 1 for the bias
            else:  # in all other hidden layers we use tanh as activation fct
                if activation_function is 0:
                    y = np.append(np.tanh(s), [1])  # We append 1 for the bias
                else:
                    if activation_function is 1:
                        y = np.append(relu(s), [1])  # We append 1 for the bias
        out = y[:-1]
        return out

    def PropagateSet(self, testset, error_function=0, activation_function=0):
        error = 0
        checked = 0
        for entry in testset.dic:
            testdata = self.convert_input(Hashable.unwrap(entry))
            targ = testset.dic[entry].reshape(9 * 9 + 1)
            if (np.sum(targ) > 0):  # We can only learn if there are actual target vectors
                targ = targ / np.linalg.norm(targ, ord=1)  # normalize (L1-norm)
                y = np.append(testdata, [1])  # We append 1 for the bias
                # Forward-propagate
                for i in range(0, self.layercount):
                    W = self.weights[i]
                    s = W.dot(y)
                    if i == self.layercount - 1:  # softmax as activationfct only in last layer
                        y = np.append(softmax(s), [1])  # We append 1 for the bias
                    else:  # in all other hidden layers we use tanh as activation fct
                        if activation_function is 0:
                            y = np.append(np.tanh(s), [1])  # We append 1 for the bias
                        else:
                            if activation_function is 1:
                                y = np.append(relu(s), [1])  # We append 1 for the bias
                # sum up the error
                if error_function is 0:
                    error += self.compute_KL_divergence(y[:-1], targ)
                else:
                    if error_function is 1:
                        error += self.compute_ms_error(y[:-1], targ)
                    else:
                        if error_function is 2:
                            error += self.compute_Hellinger_dist(y[:-1], targ)
                checked += 1
        if checked is 0:
            print("The Set contained no feasible boards")
            return
        else:
            error = error / checked  # average over training set
            return error


# test2()
