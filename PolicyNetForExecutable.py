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


class PolicyNet:
    def __init__(self):
        ### Specifications of the game
        self.n = 9  # 9x9 board

        ### Parameters of the NN
        self.eta = 0.001  # learning rate
        self.layers = [self.n * self.n, 100, 200, 300, 200, 100,
                       self.n * self.n]  # please leave the first and last equal zu n^2 for now

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

        self.errorsbyepoch = []  # mean squared error
        self.abserrorbyepoch = []  # absolute error
        self.KLdbyepoch = []  # Kullback-Leibler divergence

    ### Function Definition yard

    # activation function
    """
    def softmax(self,x):
        #Compute softmax values for each sets of scores in x.
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    """

    def compute_error(self, suggested, target):  # compare the prediction with the answer/target, absolute error
        diff = np.absolute(suggested - target)
        Error = np.inner(diff, np.ones(len(target)))
        return Error

    def compute_ms_error(self, suggested, target):  # Returns the total mean square error
        diff = np.absolute(suggested - target)
        Error = 0.5 * np.inner(diff, diff)
        return Error

    def compute_KL_divergence(self, suggested, target):  # Compute Kullback-Leibler divergence, now stable!
        t = target[
            target != 0]  # ->we'd divide by 0 else, does not have inpact on error anyway ->Problem: We don't punish the NN for predicting non-zero values on zero target!
        s = suggested[target != 0]
        diff = s / t  # this is stable
        Error = - np.inner(t * np.log(diff), np.ones(len(t)))
        return Error

    def convert_input(self, boardvector):  # rescaling help function
        boardvector = boardvector.astype(float)
        for i in range(0, len(boardvector)):
            if boardvector[i] == 0:
                boardvector[i] = 0.45
            if boardvector[i] == -1:
                boardvector[i] = -1.35
            if boardvector[i] == 1:
                boardvector[i] = 1.05
        return boardvector

    ### The actual functions

    def Propagate(self, board):
        # convert board to NeuroNet format (81-dim vector. TODO: Check for color of NN-training and flip if needed! For now, for convenience, i just subtract 0.25. We should do it similar to Heining by setting: -1.35:w, 0.45:empty, 1.05:b)
        board = board.vertices
        if len(board) != 81:
            board = board.flatten()
        for i in range(0, len(board)):
            if board[i] == 0:
                board[i] = 0.45
            if board[i] == -1:
                board[i] = -1.35
            if board[i] == 1:
                board[i] = 1.05

        y = np.append(board, [1])  # offset
        ys = [0] * self.layercount
        # Forward-propagate
        for i in range(0, self.layercount):
            W = self.weights[i]  # anders machen?
            s = W.dot(y)
            if i == self.layercount - 1:  # softmax as activationfct only in last layer
                y = np.append(softmax(s), [1])  # We append 1 for the bias
            else:  # in all other hidden layers we use tanh as activation fct
                y = np.append(np.tanh(s), [1])  # We append 1 for the bias
            ys[i] = y  # save the y values for backprop (?)
        out = y[:-1]
        return out

    def PropagateSet(self, testset):
        error = 0
        checked = 0
        for entry in testset.dic:
            testdata = self.convert_input(Hashable.unwrap(entry))
            targ = testset.dic[entry].reshape(9 * 9)
            if (np.sum(targ) > 0):  # We can only learn if there are actual target vectors
                targ = targ / np.linalg.norm(targ, ord=1)  # normalize (L1-norm)
                y = np.append(testdata, [1])
                ys = [0] * self.layercount
                # Forward-propagate
                for i in range(0, self.layercount):
                    W = self.weights[i]  # anders machen?
                    s = W.dot(y)
                    if i == self.layercount - 1:  # softmax as activationfct only in last layer
                        y = np.append(softmax(s), [1])  # We append 1 for the bias
                    else:  # in all other hidden layers we use tanh as activation fct
                        y = np.append(np.tanh(s), [1])  # We append 1 for the bias
                    ys[i] = y  # save the y values for backprop (?)
                error += self.compute_KL_divergence(y[:-1], targ)
                checked += 1
        error = error / checked  # average over training set
        return error

    def saveweights(self, folder, filename):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = dir_path + "/" + folder + "/" + filename
        np.savez(file, self.weights)

    def loadweightsfromfile(self, folder, filename):
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

