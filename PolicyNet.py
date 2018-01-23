# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 20:54:24 2018

@author: Stefan
"""
import numpy as np
import matplotlib.pyplot as plt
from Hashable import Hashable
# from TrainingDataFromSgf import TrainingDataSgfPass
import os
import time
import random
import sqlite3
from Filters import apply_filters_by_id
import collections
import io

# neccessarry to store arrays in database (from stackOverFlow)
from TrainingDataFromSgf import TrainingDataSgfPass


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)


def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


def relu(x):
    x[x < 0] = 0
    return x


class PolicyNet:
    def __init__(self, layers=[9*9, 1000, 100, 9*9+1], activation_function=0, filter_ids=[0, 2]):
        # Specifications of the game
        self.n = 9  # 9x9 board
        self.filter_ids = filter_ids
        self.filtercount = len(filter_ids)
        layers[0] += self.filtercount*self.n*self.n
        
        # Parameters of the NN
        self.layers = layers  # please leave the first and last equal zu n^2 for now
        self.activation_function = activation_function
        
        # Initialize the weights
        self.layercount = len(self.layers)-1
        self.init_weights()

        if self.activation_function is 0:
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

    def apply_filters(self, board, color=-1):
        filtered = apply_filters_by_id(board, color, self.filter_ids)
        return filtered

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

    def compute_error(self, suggested, target, error_function):
        if error_function == 0:
            return self.compute_KL_divergence(suggested, target)
        elif error_function == 1:
            return self.compute_ms_error(suggested, target)
        elif error_function == 2:
            return self.compute_Hellinger_dist(suggested, target)
        elif error_function == 3:
            return self.compute_cross_entropy(suggested, target)

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
    
    def splitintobatches(self, trainingdata, batchsize):  # splits trainingdata into batches of size batchsize
        N = len(trainingdata.dic)
        if batchsize > N:
            batchsize = N
        k = int(np.ceil(N/batchsize))

        Batch_sets = [0]*k
        Batch_sets[0] = TrainingDataSgfPass()
        Batch_sets[0].dic = dict(list(trainingdata.dic.items())[:batchsize])
        for i in range(k-1):
            Batch_sets[i] = TrainingDataSgfPass()
            Batch_sets[i].dic = dict(list(trainingdata.dic.items())[i*batchsize:(i+1)*batchsize])
        Batch_sets[k-1] = TrainingDataSgfPass()
        Batch_sets[k-1].dic = dict(list(trainingdata.dic.items())[(k-1)*batchsize:N])
        number_of_batchs = k
        return[number_of_batchs, Batch_sets]

    def extract_batches_from_db(self, db_name, batchsize, sample_proportion, duplicate=True):
        if duplicate:
            con = sqlite3.connect(r"DB's/DistributionDB's/" + db_name, detect_types=sqlite3.PARSE_DECLTYPES)
        else:
            con = sqlite3.connect(r"DB's/MoveDB's/" + db_name, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = con.cursor()
        cur.execute("select count(*) from test")
        data = cur.fetchall()
        datasize = data[0][0]
        dataprop = np.floor(float(data[0][0]) * sample_proportion)
        number_of_batches = int(np.ceil(dataprop / batchsize))
        id_set = set(range(int(datasize)) + np.ones(int(datasize), dtype='Int32'))
        batch_id_list = [0]*number_of_batches
        for i in range(number_of_batches):
            try:
                batch = set(random.sample(id_set, batchsize))
                batch_id_list[i] = batch
                id_set -= batch
            except ValueError:
                batch_id_list[i] = id_set
        batches = collections.defaultdict(dict)
        for i in range(len(batch_id_list)):
            batches[i] = collections.defaultdict(np.ndarray)
        for key in batches.keys():
            for j in batch_id_list[key]:
                cur.execute("select * from test where id = ?", (int(j),))
                data = cur.fetchone()
                batches[key][int(j)] = [data[1], data[2]]
                # TODO: Dictionaries umstellen auf (id, [board, dist])
        con.close()
        return [number_of_batches, batches]

    def saveweights(self, filename, folder='Saved_Weights'):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = dir_path + "/" + folder + "/" + filename
        np.savez(file, self.weights)
        
    def loadweightsfromfile(self, filename, folder='Saved_Weights', filter_ids=[0, 1, 2, 3, 4, 5, 6, 7]):
        # if file doesnt exist, do nothing
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = dir_path + "/" + folder + "/" + filename
        if os.path.exists(file):
            with np.load(file) as data:
                self.filter_ids = filter_ids
                self.filtercount = len(filter_ids)
                self.weights = []
                self.layer = [data['arr_0'][0].shape[1]]  # there are n+1 layers if there are n weight matrices
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

    # The actual functions

    def learn(self, trainingdata, epochs=1, eta=0.001, batch_size=10, sample_proportion=1, error_function=0, db=False,
              db_name='none', adaptive_rule='logarithmic', regularization=0):
        if adaptive_rule is "none":
            duplicate = True
        else:
            duplicate = False
        if not db:  # Dictionary Case
            [number_of_batchs, batches] = self.splitintobatches(trainingdata, batch_size)
        errors_by_epoch = []
        for epoch in range(0, epochs):
            print("current epoch: " + str(epoch))
            errors_by_epoch.append(0)
            if db:
                [number_of_batchs, batches] = self.extract_batches_from_db(db_name, batch_size, sample_proportion, duplicate)
            for i_batch in range(number_of_batchs):
                batch = batches[i_batch]
                error_in_batch = self.learn_batch(batch, eta, error_function, db, adaptive_rule=adaptive_rule)
                errors_by_epoch[epoch] += error_in_batch
            errors_by_epoch[epoch] = errors_by_epoch[epoch] / number_of_batchs
        return errors_by_epoch

    """    
    def Learnsplit(self, trainingdata, eta, batch_size, stoch_coeff, error_function, trainingrate, error_tolerance, maxepochs):
        N = len(trainingdata.dic)
        splitindex = int(round(N*trainingrate))
        trainingset, testset = TrainingDataSgfPass(), TrainingDataSgfPass()
        trainingset.dic = dict(list(trainingdata.dic.items())[:splitindex])
        testset.dic = dict(list(trainingdata.dic.items())[splitindex:])
        
        error = [error_tolerance+1]
        epochs = 0
        while error[-1:][0] > error_tolerance and epochs < maxepochs:
            epochs += 1
            self.Learn(trainingdata, 1, batch_size, stoch_coeff, error_function)
            error.append(self.PropagateSet(testset,error_function))
        return [error,epochs]
    """

    # Takes a batch, propagates all boards in that batch while accumulating delta weights. Then sums the delta weights
    # up and then adjusts the weights of the Network.
    def learn_batch(self, batch, eta_start=0.01, error_function=0,
                    db=False, adaptive_rule="linear", error_feedback=True):
        deltaweights_batch = [0] * self.layercount
        if not db:  # Dictionary case
            selection = random.sample(list(batch.dic.keys()), len(batch.dic))  # This is indeed random order.
        else:
            selection = list(batch.keys())
        batch_counter = 0
        for entry in selection:
            batch_counter += 1
            if not db:   # Usual Dictionary case. Extract input and target.
                t0 = Hashable.unwrap(entry)
                tf = self.apply_filters(t0.reshape((9, 9)))
                testdata = np.append(self.convert_input(t0),(tf))
                targ = batch.dic[entry].reshape(9*9+1)  # target output, this is to be approximated
            else:  # DB case
                t0 = batch[entry][0]
                tf = self.apply_filters(t0.reshape((9, 9)))
                testdata = np.append(self.convert_input(t0),(tf))
                targ = batch[entry][1].reshape(9 * 9 + 1)

            if np.sum(targ) > 0:  # We can only learn if there are actual target vectors
                targ_sum = np.sum(targ)  # save this for the adaptive eta
                targ = targ/np.linalg.norm(targ, ord=1)  # normalize (L1-norm)
                y = np.append(testdata, [1])  # We append 1 for the bias
                ys = [0]*self.layercount  # alocate storage space,  y_saved for backpropagation
                
                # Forward-propagate
                for i in range(0, self.layercount):
                    w = self.weights[i]
                    s = w.dot(y)
                    if i == self.layercount-1:  # softmax as activationfct only in last layer
                        y = np.append(softmax(s), [1])
                    elif self.activation_function is 0:  # in all other hidden layers we use tanh/relu as activation fct
                        y = np.append(np.tanh(s), [1])
                    elif self.activation_function is 1:
                        y = np.append(relu(s), [1])
                    ys[i] = y  # save the y values for backpropagation
                out = y
                
                # Back-propagate
                
                # Calculate Jacobian of the softmax activationfct in last layer only
                jacobian_softmax = [0] * self.layercount
                for i in range(self.layercount-1, self.layercount):
                    # Please note that I think this is pure witchcraft happening here
                    yt = ys[i]  # load y from ys and lets call it yt y_temporary
                    yt = yt[:-1]  # the last entry is from the offset, we don't need this
                    le = len(yt)
                    jacobian_softmax_temporary = np.ones((le, le))  # alloc storage temporarily
                    for j in range(0, le):
                        jacobian_softmax_temporary[j, :] *= yt[j]
                    jacobian_softmax_temporary = np.identity(le) - jacobian_softmax_temporary
                    for j in range(0, le):
                        jacobian_softmax_temporary[:, j] *= yt[j]
                    jacobian_softmax[i] = jacobian_softmax_temporary

                # Calculate Jacobian fot the not-last layers
                if self.activation_function is 0:  # Tanh
                    jacobian_tanh = [0] * self.layercount
                    for i in range(0, self.layercount):
                        yt = ys[i]  # load y from ys and lets call it yt
                        yt = yt[:-1]  # the last entry is from the offset, we don't need this
                        u = 1 - yt * yt
                        jacobian_tanh[i] = np.diag(u)
                    jacobian_hidden = jacobian_tanh
                if self.activation_function is 1:  # ReLU
                    jacobian_relu = [0]*self.layercount
                    for i in range(0, self.layercount):  # please note I think this is pure witchcraft happening here
                        yt = ys[i]  # load y from ys and lets call it yt
                        yt = yt[:-1]  # the last entry is from the offset, we don't need this
                        yt[yt > 0] = 1
                        jacobian_relu[i] = np.diag(yt)
                    jacobian_hidden = jacobian_relu
                
                # Use (L2) and (L3) to get the error signals of the layers
                errorsignals = [0] * self.layercount
                errorsignals[self.layercount-1] = jacobian_softmax[self.layercount-1]
                for i in range(2, self.layercount+1):
                    w = self.weights[self.layercount-i+1]
                    dft = jacobian_hidden[self.layercount-i]
                    errdet = np.matmul(w[:, :-1], dft)  # temporary
                    errorsignals[self.layercount-i] = np.dot(errorsignals[self.layercount-i+1], errdet)
                
                # Use (D3) to compute err_errorsignals as sum over the rows/columns? of the errorsignals weighted by
                # the deriv of the error fct by the output layer. We don't use Lemma 3 dircetly here, we just apply the
                # definition of delta_error.
                err_errorsignals = [0]*self.layercount
                if error_function is 0:
                    errorbyyzero = -targ/out[:-1]  # Kullback-Leibler divergence derivative
                elif error_function is 1:
                    errorbyyzero = out[:-1]-targ  # Mean-squared-error derivative
                elif error_function is 2:
                    errorbyyzero = 1/4*(1-np.sqrt(targ)/np.sqrt(out[:-1]))  # Hellinger-distance derivative

                for i in range(0, self.layercount):
                    err_errorsignals[i] = np.dot(errorbyyzero, errorsignals[i])  # this is the matrix variant of (D3)
                
                # Use (2.2) to get the sought derivatives. Observe that this is an outer product, though not mentioned
                # in the source (Fuck you Heining, you b*stard)
                errorbyweights = [0]*self.layercount  # dE/dW
                errorbyweights[0] = np.outer(err_errorsignals[0], testdata).T
                for i in range(1, self.layercount):
                    errorbyweights[i] = np.outer(err_errorsignals[i-1], ys[i][:-1])  # (L1)
                
                # Compute the change of weights, then apply actualization step of Gradient Descent to weight matrices
                if adaptive_rule == "linear":
                    eta = eta_start * targ_sum
                elif adaptive_rule == "logarithmic":
                    eta = eta_start * np.log(2 + targ_sum)
                elif adaptive_rule == "none":
                    eta = eta_start
                for i in range(0, self.layercount):
                    if type(deltaweights_batch[i]) is int:  # initialize
                        deltaweights_batch[i] = -eta * errorbyweights[i]
                    else:
                        deltaweights_batch[i] -= eta * errorbyweights[i]

        batch_size = batch_counter
        # TODO which eta to choose for the regularization???
        # Now adjust weights
        for i in range(0, self.layercount):
            if type(deltaweights_batch[i]) is not int:  # in this case we had no target for any board in this batch
                self.weights[i][:, :-1] = self.weights[i][:, :-1] + deltaweights_batch[i].T  # Problem: atm we only
                # adjust non-bias weights. Change that! TODO
        if error_feedback:
            error = self.propagate_set(batch, db, adaptive_rule, error_function=error_function)
            return error

    def propagate_board(self, board):
        # Convert board to NeuroNet format (82-dim vector)
        if type(board) != list and type(board) != np.ndarray:
            board = board.vertices
        if len(board) != 82:
            board = board.flatten()
        board = np.asarray(board, float)
        # Like Heining we are setting: (-1.35:w, 0.45:empty, 1.05:b)
        tf = self.apply_filters(board.reshape((9, 9)))
        for i in range(0, len(board)):
            if board[i] == np.int(0):
                board[i] = 0.45
            if board[i] == -1:
                board[i] = -1.35
            if board[i] == 1:
                board[i] = 1.05
        board = np.append(board, tf)
        y = np.append(board, [1])
        # Forward-propagate
        for i in range(0, self.layercount):
            w = self.weights[i]
            s = w.dot(y)
            if i == self.layercount-1:  # softmax as activationfct only in last layer
                y = np.append(softmax(s), [1])
            elif self.activation_function is 0:
                y = np.append(np.tanh(s), [1])
            elif self.activation_function is 1:
                y = np.append(relu(s), [1])
        out = y[:-1]
        return out
    
    def propagate_set(self, testset, db=False, adaptive_rule='linear', error_function=0):
        error = 0
        checked = 0
        if not db:
            given_set = testset.dic
        else:
            given_set = testset
        for entry in given_set:
            if not db:
                t0 = Hashable.unwrap(entry)
                tf = self.apply_filters(t0.reshape((9, 9)))
                testdata = np.append(self.convert_input(t0),(tf))
                targ = testset.dic[entry].reshape(9*9+1)
            else:
                t0 = testset[entry][0]
                tf = self.apply_filters(t0.reshape((9, 9)))
                testdata = np.append(self.convert_input(t0),(tf))
                targ = testset[entry][1].reshape(9*9+1)

            if np.sum(targ) > 0:  # We can only learn if there are actual target vectors
                targ_sum = np.sum(targ)
                targ = targ/np.linalg.norm(targ, ord=1)  # normalize (L1-norm)
                y = np.append(testdata, [1])
                # Forward-propagate
                for i in range(0, self.layercount):
                    w = self.weights[i]
                    s = w.dot(y)
                    if i == self.layercount - 1:  # softmax as activationfct only in last layer
                        y = np.append(softmax(s), [1])
                    elif self.activation_function is 0:
                        y = np.append(np.tanh(s), [1])
                    elif self.activation_function is 1:
                        y = np.append(relu(s), [1])
                # sum up the error
                error += self.compute_error(y[:-1], targ, error_function)

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
    
    # Plot results and error:
    def visualize(self, firstout, out, targ):
        games = len(out)
        f, axa = plt.subplots(3, sharex=True)
        axa[0].set_title("Error Plot")
        axa[0].set_ylabel("Mean square Error")
        axa[0].plot(range(0, games), self.errorsbyepoch)
        
        axa[1].set_ylabel("Abs Error")
        axa[1].set_ylim(0, 2)
        axa[1].plot(range(0, games), self.abserrorbyepoch)
        
        axa[2].set_xlabel("Epochs")
        axa[2].set_ylabel("K-L divergence")
        axa[2].plot(range(0, games), self.KLdbyepoch)

        plt.show()

        # Plot the results:
        plt.figure(1)
        f, axarr = plt.subplots(3, sharex=True)
        axarr[0].set_title("Neuronal Net Output Plot: First epoch vs last epoch vs target")
        axarr[0].bar(range(1, (self.n*self.n+2)), firstout)  # output of first epoch
        
        axarr[1].set_ylabel("quality in percentage")
        axarr[1].bar(range(1, (self.n*self.n+2)), out[:-1])  # output of last epoch
        
        axarr[2].set_xlabel("Board field number")
        axarr[2].bar(range(1, (self.n*self.n+2)), targ)  # target distribution

        plt.show()
    
    def visualize_error(self, errors):
        plt.plot(range(0, len(errors)), errors)
        plt.show()

        
# Tests:
def test():
    FN = PolicyNet([9*9,100,9*9+1])
    testset = TrainingDataSgfPass("dgs",'dan_data_10')
    t1 = time.time()
    err = FN.propagate_set(testset)
    print(err)
    t2 = time.time()
    print("Prop set time:",t2-t1)
    
    err_by_epoch = FN.learn(testset, epochs=1, eta=0.01, batch_size=200, error_function=0)
    t3 = time.time()
    print("Learn an epoch,", t3-t2)
    
    b=np.zeros((9,9))
    #b[1,0]=1
    #b[0,0]=-1
    sug = FN.propagate_board(b)
    print("Prop board time:", time.time()-t3)
    plt.figure(1)
    plt.bar(range(len(sug)), sug)
    plt.show()

#test()


def test1():
    FN = PolicyNet([9*9,100,9*9+1])
    FNA = PolicyNet([9*9,100,9*9+1])
    testset = TrainingDataSgfPass("dgs",range(10))
    err=FN.PropagateSet(testset)
    errA=FNA.PropagateSetAdaptive(testset)
    print(err,errA)
    
    epochs=3
    e1=FN.Learn(testset,epochs,0.01,10,1,0)
    print("first done")
    
    batch_size=10    
    [number_of_batchs, batchs] = FNA.splitintobatches(testset,batch_size)
    errors_by_epoch = []
    for epoch in range(0,epochs):
        errors_by_epoch.append(0)
        for i_batch in range(0,number_of_batchs):
            error_in_batch = FNA.LearnSingleBatchAdaptive(batchs[i_batch], 0.01, 1, 0)
            errors_by_epoch[epoch] += error_in_batch
        errors_by_epoch[epoch] = errors_by_epoch[epoch] / number_of_batchs
    e2=errors_by_epoch
    
    b=np.zeros((9,9))
    s1=FN.Propagate(b)
    s2=FNA.Propagate(b)
    
    plt.figure(0)
    plt.bar(range(len(s1)),s1)
    plt.figure(1)
    plt.bar(range(len(s2)),s2)
    
    print(e1,e2)
    
# test1()


def test2():
    FN = PolicyNet([9*9,1000,200,9*9+1], filter_ids=[0,1,2,3,4,5,6,7])
    testset = TrainingDataSgfPass("dgs","dan_data_10")
    print("games imported")
    batch_size = 50
    eta = 0.001
    err_fct = 0
    [number_of_batchs, batchs] = FN.splitintobatches(testset,batch_size)
    print("split up into",number_of_batchs,"with size",batch_size)
    errors_by_epoch = []
    start = time.time()
    epoch = 0
    while time.time()-start < 8.5*60*60:
    #for epoch in range(0,epochs):
        t=time.time()
        errors_by_epoch.append(0)
        for i_batch in range(0,number_of_batchs):
            error_in_batch = FN.learn_batch(batchs[i_batch], eta, 0, False, "logarithmic", True)
            errors_by_epoch[epoch] += error_in_batch
        errors_by_epoch[epoch] = errors_by_epoch[epoch] / number_of_batchs
        print("epoch",epoch,"error",errors_by_epoch[epoch])
        print(np.round(time.time()-t))
        epoch=epoch+1
    FN.saveweights("ambitestfilt1234567logrule_111")
    print("total time:",time.time()-start,"and epochs",epoch)
    print("final error:",errors_by_epoch)
    plt.plot(range(0,len(errors_by_epoch)),errors_by_epoch)

#test2()


def test3():
    FN = PolicyNet([9*9,1000,200,9*9+1])
    FN.loadweightsfromfile("ambtestfilt")
    testset = TrainingDataSgfPass("dgs",range(30))
    b1=np.zeros((9,9))
    err = FN.propagate_set(testset)
    print(err)
    s1=FN.propagate_board(b1)
    b2=b1
    b2[0,1]=1
    b2[0,0]=-1
    s2=FN.propagate_board(b2)
    plt.figure(0)
    plt.bar(range(len(s1)),s1)
    plt.figure(1)
    plt.bar(range(len(s2)),s2)
#test3()

def batch_extraction_test():
    net = PolicyNet()
    sample_prop = 1
    [no, batches] = net.extract_batches_from_db('dan_data_10', 100, sample_proportion=sample_prop)
    data = batches[1].keys()
    print(no)
    print(data)
#batch_extraction_test()

def test4():
    con = sqlite3.connect(r"DB's/DistributionDB's/dan_data_10_topped_up", detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("select * from dist where id = ?", (3,))
    data = cur.fetchone()
    print(data[1])
#test4()

def test5():  # vgl dbs mit dicts
    PN = PolicyNet()
    sample_prop = 1
    batch_size = 100

    t = time.time()
    dictset = TrainingDataSgfPass("dgs", "dan_data_10")
    t1 = time.time()
    print("dict import time:", t1 - t)
    [dict_n, dict_batches] = PN.splitintobatches(dictset, batch_size)
    t2 = time.time()
    print("dict split time:", t2-t1)

    [db_n, db_batches] = PN.extract_batches_from_db('dan_data_10', batch_size, sample_proportion=sample_prop)
    t3 = time.time()
    print("db import and split time:", t3 - t2)

    #data = db_batches[1].keys()
    print("DB batches: ", db_n)
    print("Dict batches: ", dict_n)
    #print(data)
#test5()

def test6():
    PN = PolicyNet()
    sample_prop = 1
    batch_size = 100
    [db_n, db_batches] = PN.extract_batches_from_db('dan_data_10', batch_size, sample_proportion=sample_prop, duplicate=False)
    print(db_n)

    db_name = 'dan_data_10'
    con = sqlite3.connect(r"DB's/MoveDB's/" + db_name, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("select count(*) from test")
    data = cur.fetchall()
    datasize = data[0][0]
    print(np.ceil(datasize/100))

#test6()

def test7():  # Test the error of the set on a db-fan file
    PN = PolicyNet(filter_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    PN.loadweightsfromfile("ambitestfilt1234567logrule", filter_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    db_name = "dan_data_295"
    con = sqlite3.connect(r"DB's/MoveDB's/" + db_name, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("select count(*) from test")
    data = cur.fetchall()
    con.close()
    datasize = data[0][0]

    [_, whole_set_temp] = PN.extract_batches_from_db(db_name, datasize, 1, duplicate=False)
    db_dan_295 = whole_set_temp[0]

    print("Games have been imported from ", db_name)

    error = PN.propagate_set(db_dan_295, True, "logarithmic", 0)
    print(error)

#test7()
