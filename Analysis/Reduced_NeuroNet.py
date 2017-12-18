"""
Created on Wed Nov  8 11:00:05 2017
@author: Stefan Peidli
License: MIT
Tags: Neural Network
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def relu(x):
    x[x < 0] = 0
    return x

class Habitat():
    def __init__(self, layers, pop_size = 10, geno_size = 4):
        self.layers = layers
        self.genomes = []
        self.pop_size = pop_size
        for i in range(geno_size):
            self.genomes.append(Genome(layers, i))
        self.status = "dead"
        self.population = []

    def set_alive(self):
        self.status="alive"
        for i in range (self.pop_size):
            genotype = random.choice(self.genomes)
            #self.population.append(NeuroNet(self.layers[0],self.layers[-1],self.layers[1:-1],genotype.act_fct,genotype.last_act_fct, genotype.error_fct))
            self.population.append(Individual(genotype))

    def set_dead(self):
        self.status="dead"
        self.population = []

class Genome:
    #An indiviual has a unique Genome. The Genome contains all parameters describing a Neural Network.
    def __init__(self, layers, identifier):
        #Genome
        self.eta = random.choice([1,0.1,0.01,0.001,0.0001,0])
        self.batch_size = random.choice([1,2,3])
        self.stochasticity = random.choice([0,1]) #1: stoch grad desc, 0: vanilla grad desc
        self.error_fct = 1
        self.act_fct = 0
        self.last_act_fct = 1
        self.identifier = identifier
        #part of genome?:
        self.layers =layers

class Individual:
    def __init__(self, Genome):
        self.Genome = Genome
        self.NN = NeuroNet(Genome.layers[0],Genome.layers[-1],Genome.layers[1:-1],Genome.act_fct,Genome.last_act_fct,Genome.error_fct)


class NeuroNet:
    def __init__(self, input_size=1, output_size=1, hidden_layers=[5, 5], act_fct=0, last_act_fct=1, error_function=1):
        ### Specifications of task
        self.n = 9  # 9x9 board
        layers = [input_size, *hidden_layers, output_size]

        ### Parameters of the NN
        self.layers = layers  # all layers of the NN
        self.act_fct = act_fct # activation function of the first and hidden layers
        self.last_act_fct = last_act_fct # activation fct of the last layers

        ### Initialize the weights
        self.layercount = len(self.layers) - 1
        self.init_weights()

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

    # error fct Number 4
    def compute_experimental(self, suggested, target, gamma):
        alpha = 1 / gamma
        beta = np.log(gamma)
        error = alpha * np.sum(np.exp((suggested - target) * beta))
        return error

    def compute_experimental_gradient(self, suggested, target, gamma):
        alpha = 1 / gamma
        beta = np.log(gamma)
        gradient = alpha * beta * np.exp((suggested - target) * beta)
        return gradient

    # error fct Number x, actually not a good one. Only for statistics
    def compute_abs_error(self, suggested, target):  # compare the prediction with the answer/target, absolute error
        difference = np.absolute(suggested - target)
        Error = np.inner(difference, np.ones(len(target)))
        return Error

    def compute_entropy(self, distribution):
        return -np.inner(distribution, np.log(distribution))

    ### Auxilary and Utilary Functions

    def init_weights(self):
        ### Initialize the weights
        # Xavier Initialization
        if self.act_fct is 0:  # TODO wie im last layer haben wir softmax! wie initialisieren???
            mu = 0
            self.weights = [0] * self.layercount  # alloc memory
            for i in range(0, self.layercount):
                sigma = 1 / np.sqrt(self.layers[i + 1])
                self.weights[i] = np.random.normal(mu, sigma, (self.layers[i + 1], self.layers[i] + 1))
        # He Initialization
        elif self.act_fct is 1:
            mu = 0
            self.weights = [0] * self.layercount  # alloc memory
            for i in range(0, self.layercount):
                sigma = np.sqrt(2) / np.sqrt(
                    self.layers[i + 1])  # vgl Heining #TODO:Check if inputs are well-behaved (approx. normalized)
                self.weights[i] = np.random.normal(mu, sigma, (
                    self.layers[i + 1], self.layers[i] + 1))  # edit: the +1 in the input dimension is for the bias

    def weight_ensemble(self, testset, instances = 1, details = True):
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
        improvement = np.round((first_value - optimal_value)/first_value,4)
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
        N = len(trainingdata[0])
        if batchsize > N:
            batchsize = N
        k = int(np.ceil(N / batchsize))
        permutation = np.arange(0,N)
        random.shuffle(permutation)
        trainingdata = [trainingdata[0][permutation],trainingdata[1][permutation]]
        Batch_sets = [0] * k
        for i in range(k - 1):
            Batch_sets[i]=[trainingdata[0][i*batchsize:(i+1)*batchsize],trainingdata[1][i*batchsize:(i+1)*batchsize]]
        Batch_sets[-1] = [trainingdata[0][(k-1) * batchsize:N],trainingdata[1][(k-1) * batchsize:N]]
        number_of_batchs = k
        return [number_of_batchs, Batch_sets]

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

    ### The actual functions

    def Learn(self, trainingdata, epochs=1, eta=0.01, batch_size=1, stoch_coeff=1, error_function=1):
        if stoch_coeff==0:#no stoch grad descent
            [number_of_batchs, batchs] = self.splitintobatches(trainingdata, batch_size)
        errors_by_epoch = []
        for epoch in range(0, epochs):
            if stoch_coeff == 1:
                [number_of_batchs, batchs] = self.splitintobatches(trainingdata, batch_size)
            errors_by_epoch.append(0)
            for i_batch in range(0, number_of_batchs):
                error_in_batch = self.LearnSingleBatch(batchs[i_batch], eta, stoch_coeff, error_function)
                errors_by_epoch[epoch] += error_in_batch
            errors_by_epoch[epoch] = errors_by_epoch[epoch] / number_of_batchs
        return errors_by_epoch

    def Learnsplit(self, trainingdata, eta, batch_size, stoch_coeff, error_function, trainingrate, error_tolerance,
                   maxepochs):
        N = len(trainingdata.dic)
        splitindex = int(round(N * trainingrate))
        trainingset, testset = TrainingData(), TrainingData()  # TODO: check if this is fine with TraingDataSgf method
        trainingset.dic = dict(list(trainingdata.dic.items())[:splitindex])
        testset.dic = dict(list(trainingdata.dic.items())[splitindex:])

        error = [error_tolerance + 1]
        epochs = 0
        while error[-1:][0] > error_tolerance and epochs < maxepochs:
            epochs += 1
            self.Learn(trainingdata, 1, batch_size, stoch_coeff, error_function)
            error.append(self.PropagateSet(testset, error_function))
        return [error, epochs]

    def LearnSingleBatch(self, batch, eta=0.01, stoch_coeff=1,error_function=0):
        # takes a batch, propagates all boards in that batch while accumulating deltaweights. Then sums the deltaweights
        #  up and the adjustes the weights of the Network.
        deltaweights_batch = [0] * self.layercount
        #selection_size = int(np.round(len(batch[0]) * stoch_coeff))
        #if selection_size == 0:  # prevent empty selection
        #    selection_size = 1
        #random_selection = random.sample(np.arange(len(batch[0])), selection_size)
        for entry in range(0,len(batch[0])):
            testdata = batch[0][entry]  # input
            targ = batch[1][entry]  # target output, this is to be approximated
            y = np.append(testdata, [1])  # We append 1 for the bias
            ys = [0] * self.layercount  # y_saved for backpropagation
            # Forward-propagate
            for i in range(0, self.layercount):
                W = self.weights[i]  # anders machen?
                s = W.dot(y)
                if i == self.layercount - 1:  # softmax as activationfct only in last layer
                    if self.last_act_fct == 0:
                        y = np.append(softmax(s), [1])  # We append 1 for the bias
                    elif self.last_act_fct == 1:
                        #print("s",s)
                        y = np.append(np.tanh(s), [1])
                        #print(y)
                elif self.act_fct is 0:
                    y = np.append(np.tanh(s), [1])  # We append 1 for the bias
                elif self.act_fct is 1:
                    y = np.append(relu(s), [1])  # We append 1 for the bias
                ys[i] = y  # save the y values for backpropagation
            out = y

            # Backpropagation

            if self.last_act_fct == 0:
                i = self.layercount-1
                yt = ys[i]  # load y from ys and lets call it yt y_temporary
                yt = yt[:-1]  # the last entry is from the offset, we don't need this
                le = len(yt)
                Jacobian_Softmax_temporary = np.ones((le, le))  # alloc storage temporarily
                for j in range(0, le):
                    Jacobian_Softmax_temporary[j, :] *= yt[j]
                Jacobian_Softmax_temporary = np.identity(le) - Jacobian_Softmax_temporary
                for j in range(0, le):
                    Jacobian_Softmax_temporary[:, j] *= yt[j]
                Jacobian_Softmax = Jacobian_Softmax_temporary
                Jacobian_last = Jacobian_Softmax
                # Jacobian_Softmax is quadratic and symmetric.
            elif self.last_act_fct == 1:
                i = self.layercount-1
                yt = ys[i]  # load y from ys and lets call it yt
                yt = yt[:-1]  # the last entry is from the offset, we don't need this
                u = 1 - yt * yt
                Jacobian_last = np.diag(u)

            if self.act_fct is 0:
                # Calc Jacobian of tanh
                Jacobian_tanh = [0] * self.layercount
                for i in range(0,
                               self.layercount):  # please note that I think this is pure witchcraft happening here
                    yt = ys[i]  # load y from ys and lets call it yt
                    yt = yt[:-1]  # the last entry is from the offset, we don't need this
                    u = 1 - yt * yt
                    Jacobian_tanh[i] = np.diag(u)
                Jacobian_hidden = Jacobian_tanh
            if self.act_fct is 1:
                # Calc Jacobian of relu
                Jacobian_relu = [0] * self.layercount
                for i in range(0,
                               self.layercount):  # please note that I think this is pure witchcraft happening here
                    yt = ys[i]  # load y from ys and lets call it yt
                    yt = yt[:-1]  # the last entry is from the offset, we don't need this
                    yt[
                        yt > 0] = 1  # actually 0 values go to 1 also. this is not so easy, thus I leave it like that for now
                    Jacobian_relu[i] = np.diag(yt)
                Jacobian_hidden = Jacobian_relu

            # Use (L2) and (L3) to get the error signals of the layers
            errorsignals = [0] * self.layercount
            errorsignals[self.layercount - 1] = Jacobian_last # (L2), the error signal of the output layer
            for i in range(2, self.layercount + 1):
                w = self.weights[self.layercount - i + 1]
                DFt = Jacobian_hidden[self.layercount - i]  # tanh
                errdet = np.matmul(w[:, :-1], DFt)  # temporary
                errorsignals[self.layercount - i] = np.dot(errorsignals[self.layercount - i + 1],errdet)  # (L3)

            # Use (D3) to compute err_errorsignals as sum over the rows/columns? of the errorsignals weighted by the
            #  deriv of the error fct by the output layer. We don't use Lemma 3 dircetly here, we just apply the
            # definition of delta_error
            err_errorsignals = [0] * self.layercount
            if error_function is 0:
                errorbyyzero = -targ / out[:-1]  # Kullback-Leibler divergence derivative
            elif error_function is 1:
                errorbyyzero = out[:-1] - targ  # Mean-squared-error derivative
            elif error_function is 2:
                errorbyyzero = 1 / 4 * (1 - np.sqrt(targ) / np.sqrt(out[:-1]))  # Hellinger-distance derivative
            elif error_function is 3:
                errorbyyzero = self.compute_experimental_gradient(out[:-1], targ, 1000)
            # errorbyyzero = self.chosen_error_fct(targ,out)
            for i in range(0, self.layercount):
                err_errorsignals[i] = np.dot(errorbyyzero, errorsignals[i])  # this is the matrix variant of D3

            # Use (2.2) to get the sought derivatives. Observe that this is an outer product.
            errorbyweights = [0] * self.layercount  # dE/dW
            errorbyweights[0] = np.outer(err_errorsignals[0], testdata).T  # Why do I need to transpose here???
            for i in range(1, self.layercount):
                errorbyweights[i] = np.outer(err_errorsignals[i - 1], ys[i][:-1])  # (L1)

            # Compute the change of weights, that means, then apply actualization step of Gradient Descent to weight matrices
            for i in range(0, self.layercount):
                if type(deltaweights_batch[i]) is int:  # initialize
                    deltaweights_batch[i] = -eta * errorbyweights[i]
                else:
                    deltaweights_batch[i] -= eta * errorbyweights[i]

        # now adjust weights
        for i in range(0, self.layercount):
            if type(deltaweights_batch[
                        i]) is not int:  # in this case we had no target for any board in this batch
                self.weights[i][:, :-1] = self.weights[i][:, :-1] + deltaweights_batch[
                    i].T  # Problem: atm we only adjust non-bias weights. Change that!
        error = self.PropagateSet(batch, error_function)
        return error


    def Propagate(self, input):
        y = np.append(input, [1])  # apply offset
        # Forward-propagate
        for i in range(0, self.layercount):
            W = self.weights[i]
            s = W.dot(y)
            if i == self.layercount - 1:  # softmax as activationfct only in last layer
                if self.last_act_fct == 0:
                    y = np.append(softmax(s), [1])  # We append 1 for the bias
                elif self.last_act_fct == 1:
                    y = np.append(np.tanh(s), [1])
            elif self.act_fct is 0:
                y = np.append(np.tanh(s), [1])  # We append 1 for the bias
            elif self.act_fct is 1:
                y = np.append(relu(s), [1])  # We append 1 for the bias
        out = y[:-1]
        return out

    def PropagateSet(self, testset, error_function=1):
        error = 0
        checked = 0
        for entry in range(0,len(testset[0])):
            testdata = testset[0][entry]  # input
            targ = testset[1][entry]  # target output, this is to be approximated
            y = np.append(testdata, [1])  # We append 1 for the bias
            # Forward-propagate
            for i in range(0, self.layercount):
                W = self.weights[i]
                s = W.dot(y)
                if i == self.layercount - 1:  # softmax as activationfct only in last layer
                    if self.last_act_fct == 0:
                        y = np.append(softmax(s), [1])  # We append 1 for the bias
                    elif self.last_act_fct == 1:
                        y = np.append(np.tanh(s), [1])
                elif self.act_fct is 0:
                    y = np.append(np.tanh(s), [1])  # We append 1 for the bias
                elif self.act_fct is 1:
                    y = np.append(relu(s), [1])  # We append 1 for the bias
            # sum up the error
            if error_function is 0:
                error += self.compute_KL_divergence(y[:-1], targ)
            elif error_function is 1:
                error += self.compute_ms_error(y[:-1], targ)
            elif error_function is 2:
                error += self.compute_Hellinger_dist(y[:-1], targ)
            elif error_function is 3:
                error += self.compute_experimental(y[:-1], targ, 1000)
            checked += 1
        if checked is 0:
            print("The Set contained no feasible input")
            return
        else:
            error = error / checked  # average over training set
            return error

    ### Plot results and error:
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

        # Plot the results:
        plt.figure(1)
        f, axarr = plt.subplots(3, sharex=True)
        axarr[0].set_title("Neuronal Net Output Plot: First epoch vs last epoch vs target")
        axarr[0].bar(range(1, (self.n * self.n + 2)), firstout)  # output of first epoch

        axarr[1].set_ylabel("quality in percentage")
        axarr[1].bar(range(1, (self.n * self.n + 2)), out[:-1])  # output of last epoch

        axarr[2].set_xlabel("Board field number")
        axarr[2].bar(range(1, (self.n * self.n + 2)), targ)  # target distribution

    def visualize_error(self, errors):
        plt.plot(range(0, len(errors)), errors)


def test():
    a=1.35
    b=-1.05
    PN = NeuroNet()
    input=np.array([a,b,a,b,b,a,b])
    output=np.array([1,0,1,0,0,1,0])
    testset = [input,output]

    imp=PN.weight_ensemble(testset,50,True)
    print("Improved initial error by",imp*100,"%.")

    out1=[]
    for i in input:
        out1.append(PN.Propagate(i))

    epochs = 1000
    eta=0.01

    error_by_epoch = PN.Learn(testset, epochs, eta,2,1,1)
    #print(error_by_epoch)

    plt.figure(0)
    plt.plot(range(0, epochs), error_by_epoch)
    plt.show()

    out=[]
    for i in input:
        out.append(PN.Propagate(i))

    plt.figure(1)
    plt.plot(range(0, len(out)), out,'b')
    plt.plot(range(0, len(out1)), out1,'r')
    plt.plot(range(0, len(output)), relu(output),'g')
    #plt.show()
    print("Final Error:",error_by_epoch[-1])

#test()

def test2():
    layers = [1, 5, 5, 1]
    pop_size = 100
    genome_size = 20
    competition = np.ceil(pop_size * 0.5)

    #testproblem
    a = 1.35
    b = -1.05
    input = np.array([a, b, a, b, b, a, b])
    output = np.array([1, 0, 1, 0, 0, 1, 0])
    testset = [input, output]

    #init
    H=Habitat(layers, pop_size, genome_size)
    H.set_alive()

    #evolve:
    for era in range(20):
        plt.figure(era)
        score = []
        for indiv in H.population:
            indiv.NN.weight_ensemble(testset,10,False)
            error = indiv.NN.Learn(testset,50,indiv.Genome.eta)

            score.append(indiv.NN.PropagateSet(testset))

            plt.plot(range(len(error)),error)

        ranking = np.argsort(score)
        print("era",era)
        print("Ranking:",ranking[:int(competition)])
        winner = np.argmin(score)
        print("Winner:", winner)

        best3 = ranking[:int(competition)]
        H.genomes = []
        for w in best3:
            H.genomes.append(H.population[w].Genome)

        #new era
        H.set_dead()
        H.set_alive()

    survivours = []
    for g in H.genomes:
        survivours.append(g.identifier)
    print("survivours:",survivours)

    #plt.show()

test2()