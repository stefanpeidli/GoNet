"""
Created on Wed Nov  8 11:00:05 2017
@author: Stefan Peidli
License: MIT
Tags: Policy-net, Neural Network
"""
import numpy as np
import matplotlib.pyplot as plt
from Hashable import Hashable
from TrainingDataFromSgf import TrainingData #legacy
from TrainingDataFromSgf import TrainingDataSgfPass #82 not 81
import os
import time
import random

def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
def relu(x):
    x[x<0]=0
    return x

class PolicyNet:
    def __init__(self,layers=[9*9,1000,100,9*9+1],activation_function=0,error_function=0):
        ### Specifications of the game
        self.n=9 # 9x9 board
        
        ### Parameters of the NN
        self.eta = 0.001 # learning rate
        self.layers = layers #please leave the first and last equal zu n^2 for now
        self.activation_function=activation_function
        
        ### Initialize the weights
        # here by a normal distribution N(mu,sigma)
        self.layercount = len(self.layers)-1
        if self.activation_function is 0:#TODO wie im last layer haben wir softmax! wie initialisieren???
            mu=0
            self.weights=[0]*self.layercount #alloc memory
            for i in range(0,self.layercount):
                sigma = 1/np.sqrt(self.layers[i+1]) #vgl Heining #TODO:Check if inputs are well-behaved (approx. normalized)
                self.weights[i]=np.random.normal(mu, sigma, (self.layers[i+1],self.layers[i]+1))#edit: the +1 in the input dimension is for the bias
            
        elif self.activation_function is 1:
            mu=0
            self.layercount = len(self.layers)-1
            self.weights=[0]*self.layercount #alloc memory
            for i in range(0,self.layercount):
                sigma = np.sqrt(2)/np.sqrt(self.layers[i+1]) #vgl Heining #TODO:Check if inputs are well-behaved (approx. normalized)
                self.weights[i]=np.random.normal(mu, sigma, (self.layers[i+1],self.layers[i]+1))#edit: the +1 in the input dimension is for the bias
            
        
        
        ### Alloc memory for the error statistics
        #Hint:b=a[:-1,:] this is a handy formulation to delete the last column, Thus ~W=W[:-1,1]
        
        #self.errorsbyepoch=[] #mean squared error
        #self.abserrorbyepoch=[] #absolute error
        #self.KLdbyepoch=[] #Kullback-Leibler divergence
        
        
    ### Function Definition yard
      
    # error functions

    #error fct Number 0
    def compute_KL_divergence (self, suggested, target): #Compute Kullback-Leibler divergence, now stable!
        t=target[target!=0] #->we'd divide by 0 else, does not have inpact on error anyway ->Problem: We don't punish the NN for predicting non-zero values on zero target!
        s=suggested[target!=0]
        difference=s/t #this is stable
        Error = - np.inner(t*np.log(difference),np.ones(len(t)))
        return Error
    
    #error fct Number 1
    def compute_ms_error (self, suggested, target): #Returns the total mean square error
        difference = np.absolute(suggested - target)
        Error = 0.5*np.inner(difference,difference)
        return Error
    
    #error fct Number 2
    def compute_Hellinger_dist (self, suggested, target):
        return np.linalg.norm(np.sqrt(suggested)-np.sqrt(target), ord=2) /np.sqrt(2)
    
    #error fct Number 3
    def compute_cross_entropy (self, suggested, target):
        return self.compute_entropy(target) + self.compute_KL_divergence(target,suggested) #wie rum kldiv?  
    
    #error fct Number 4
    def compute_experimental (self, suggested, target, gamma):
        alpha = 1/gamma
        beta = np.log(gamma)
        error = alpha*np.sum(np.exp((suggested - target)*beta))
        return error
    
    def compute_experimental_gradient(self, suggested, target, gamma):
        alpha = 1/gamma
        beta = np.log(gamma)
        gradient = alpha*beta*np.exp((suggested - target)*beta)
        return gradient
        
    #error fct Number x, actually not a good one. Only for statistics
    def compute_abs_error (self, suggested, target): #compare the prediction with the answer/target, absolute error
        difference = np.absolute(suggested - target)
        Error = np.inner(difference,np.ones(len(target)))
        return Error
    
    def compute_entropy (self, distribution):
        return -np.inner(distribution,np.log(distribution))
    
    ### Auxilary and Utilary Functions
    
    def convert_input(self, boardvector):#rescaling help function [-1,0,1]->[-1.35,0.45,1.05]
        boardvector=boardvector.astype(float)
        for i in range(0,len(boardvector)):
            if boardvector[i]==0:
                boardvector[i] = 0.45
            if boardvector[i]==-1:
                boardvector[i] = -1.35
            if boardvector[i]==1:
                boardvector[i] = 1.05
        return boardvector
    
    def splitintobatches(self, trainingdata, batchsize): #splits trainingdata into batches of size batchsize
        N = len(trainingdata.dic)
        if batchsize > N:
            batchsize = N
        k = int(np.ceil(N/batchsize))
        
        Batch_sets=[0]*k
        Batch_sets[0]=TrainingData() #TODO: check if this is fine with TraingDataSgf method
        Batch_sets[0].dic = dict(list(trainingdata.dic.items())[:batchsize])
        for i in range(k-1):
            Batch_sets[i]=TrainingData()
            Batch_sets[i].dic=dict(list(trainingdata.dic.items())[i*batchsize:(i+1)*batchsize])
        Batch_sets[k-1]=TrainingData()
        Batch_sets[k-1].dic = dict(list(trainingdata.dic.items())[(k-1)*batchsize:N])
        number_of_batchs = k
        return[number_of_batchs, Batch_sets]
        
    def saveweights(self, filename, folder = 'Saved_Weights'):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = dir_path + "/" + folder + "/" + filename
        np.savez(file,self.weights)
        
    def loadweightsfromfile(self, filename, folder = 'Saved_Weights'):
        # if file doesnt exist, do nothing
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = dir_path + "/" + folder + "/" + filename
        if os.path.exists(file):
            with np.load(file) as data:
                self.weights=[]
                self.layer=[data['arr_0'][0].shape[1]] #there are n+1 layers if there are n weightmatrices
                for i in range(len(data['arr_0'])):
                    self.weights.append(data['arr_0'][i])
                    tempshape=data['arr_0'][i].shape
                    self.layer.append(tempshape[0])
                self.layercount=len(self.layer)-1
        elif os.path.exists(file + ".npz"):
            with np.load(file + ".npz") as data:
                self.weights=[]
                for i in range(len(data['arr_0'])):
                    self.weights.append(data['arr_0'][i])
    
    
    ### The actual functions

    def Learn(self, trainingdata, epochs=1, eta=0.01, batch_size=1, stoch_coeff=1, error_function=1):
        if batch_size is 1:
            print("Minibatch mode activated")
            errors_by_epoch = []
            for epoch in range(0,epochs):
                error = self.LearnwithMiniBatch(trainingdata, eta, stoch_coeff, error_function)
                errors_by_epoch.append(error)
            return errors_by_epoch
        else:
            [number_of_batchs, batchs] = self.splitintobatches(trainingdata,batch_size)
            errors_by_epoch = []
            for epoch in range(0,epochs):
                errors_by_epoch.append(0)
                for i_batch in range(0,number_of_batchs):
                    error_in_batch = self.LearnSingleBatch(batchs[i_batch], eta, stoch_coeff, error_function)
                    errors_by_epoch[epoch] += error_in_batch
                errors_by_epoch[epoch] = errors_by_epoch[epoch] / number_of_batchs
            return errors_by_epoch
        
    def Learnsplit(self, trainingdata, eta, batch_size, stoch_coeff, error_function, trainingrate, error_tolerance, maxepochs):
        N = len(trainingdata.dic)
        splitindex = int(round(N*trainingrate))
        trainingset, testset = TrainingData(), TrainingData() #TODO: check if this is fine with TraingDataSgf method
        trainingset.dic = dict(list(trainingdata.dic.items())[:splitindex])
        testset.dic = dict(list(trainingdata.dic.items())[splitindex:])
        
        error = [error_tolerance+1]
        epochs = 0
        while error[-1:][0] > error_tolerance and epochs < maxepochs:
            epochs += 1
            self.Learn(trainingdata, 1, batch_size, stoch_coeff, error_function)
            error.append(self.PropagateSet(testset,error_function))
        return [error,epochs]

    
    
    def LearnwithMiniBatch(self, trainingdata, eta = 0.01, stoch_coeff = 1, error_function = 1):
        random_selection = random.sample(list(trainingdata.dic.keys()),int(np.round(len(trainingdata.dic)*stoch_coeff)))
        for entry in random_selection:
            testdata = self.convert_input(Hashable.unwrap(entry))
            targ = trainingdata.dic[entry].reshape(9*9+1)
            if(np.sum(targ)>0): #We can only learn if there are actual target vectors
                targ = targ/np.linalg.norm(targ, ord=1) #normalize (L1-norm)
                y = np.append(testdata,[1])
                ys = [0]*self.layercount
                #Forward-propagate
                for i in range(0,self.layercount): 
                    W = self.weights[i] #anders machen?
                    s = W.dot(y)
                    if i==self.layercount-1: #softmax as activationfct only in last layer    
                        y = np.append(softmax(s),[1]) #We append 1 for the bias
                    else: #in all other hidden layers we use tanh as activation fct
                        if self.activation_function is 0:
                            y = np.append(np.tanh(s),[1]) #We append 1 for the bias
                        else: 
                            if self.activation_function is 1:
                                y = np.append(relu(s),[1]) #We append 1 for the bias
                    ys[i]=y #save the y values for backprop (?)
                out=y
                
                #Backpropagation
                
                #Calculate Jacobian of the softmax activationfct in last layer 
                Jacobian_Softmax = [0]*self.layercount
                for i in range(self.layercount-1,self.layercount): #please note that I think this is pure witchcraft happening here
                    yt=ys[i] #load y from ys and lets call it yt
                    yt=yt[:-1] #the last entry is from the offset, we don't need this
                    le=len(yt)
                    DFt=np.ones((le,le)) #alloc storage temporarily
                    for j in range(0,le):
                        DFt[j,:]*=yt[j]
                    DFt=np.identity(le) - DFt
                    for j in range(0,le):
                        DFt[:,j]*=yt[j]
                    Jacobian_Softmax[i]=DFt
                #A Jacobian is quadratic and symmetric
            
                if self.activation_function is 0:
                    #Calc Jacobian of tanh
                    Jacobian_tanh = [0]*self.layercount
                    for i in range(0,self.layercount): #please note that I think this is pure witchcraft happening here
                        yt=ys[i] #load y from ys and lets call it yt
                        yt=yt[:-1] #the last entry is from the offset, we don't need this
                        u=1-yt*yt
                        Jacobian_tanh[i]=np.diag(u)
                    Jacobian_hidden = Jacobian_tanh
                if self.activation_function is 1:
                    #Calc Jacobian of relu
                    Jacobian_relu = [0]*self.layercount
                    for i in range(0,self.layercount): #please note that I think this is pure witchcraft happening here
                        yt=ys[i] #load y from ys and lets call it yt
                        yt=yt[:-1] #the last entry is from the offset, we don't need this
                        yt[yt>0]=1#actually 0 values go to 1 also. this is not so easy, thus I leave it like that for now
                        Jacobian_relu[i]=np.diag(yt)
                    Jacobian_hidden = Jacobian_relu
                
                #Use (L2) and (L3) to get the error signals of the layers
                errorsignals = [0]*self.layercount
                errorsignals[self.layercount-1] = Jacobian_Softmax[self.layercount-1] # (L2), the error signal of the output layer can be computed directly, here we actually use softmax
                for i in range(2,self.layercount+1):
                    w = self.weights[self.layercount-i+1]
                    DFt = Jacobian_hidden[self.layercount-i] #tanh
                    errdet = np.matmul(w[:,:-1],DFt) #temporary #TODO: Jacobi für tanh und relu ist vector, use that
                    errorsignals[self.layercount-i] = np.dot(errorsignals[self.layercount-i+1],errdet) # (L3), does python fucking get that?
                
                #Use (D3) to compute err_errorsignals as sum over the rows/columns? of the errorsignals weighted by the deriv of the error fct by the output layer. We don't use Lemma 3 dircetly here, we just apply the definition of delta_error
                err_errorsignals = [0]*self.layercount
                if error_function is 0:
                    errorbyyzero = -targ/out[:-1] #Kullback-Leibler divergence derivative
                else:
                    if error_function is 1:
                        errorbyyzero = out[:-1]-targ #Mean-squared-error derivative
                    else:
                        if error_function is 2:
                            errorbyyzero = 1/4*(1-np.sqrt(targ)/np.sqrt(out[:-1])) #Hellinger-distance derivative
                        else:
                            if error_function is 3:
                                errorbyyzero=self.compute_experimental_gradient(out[:-1],targ,1000)
                #errorbyyzero = self.chosen_error_fct(targ,out)
                for i in range(0,self.layercount):
                    err_errorsignals[i] = np.dot(errorbyyzero,errorsignals[i]) #this is the matrix variant of D3
                
                #Use (2.2) to get the sought derivatives. Observe that this is an outer product, though not mentioned in the source (Fuck you Heining, you bastard)
                errorbyweights = [0]*self.layercount #dE/dW
                errorbyweights[0] = np.outer(err_errorsignals[0],testdata).T #Why do I need to transpose here???
                for i in range(1,self.layercount): 
                    errorbyweights[i] = np.outer(err_errorsignals[i-1],ys[i][:-1]) # (L1)
                
                #Compute the change of weights, that means, then apply actualization step of Gradient Descent to weight matrices
                deltaweights=[0]*self.layercount
                for i in range(0,self.layercount):
                    deltaweights[i] =-eta*errorbyweights[i]
                    self.weights[i][:,:-1]= self.weights[i][:,:-1]+ deltaweights[i].T #TODO: Problem: atm we only adjust non-bias weights. Change that!
        
        error = self.PropagateSet(trainingdata, error_function)
        
        return error
    
    def LearnSingleBatch(self, batch, eta=0.01, stoch_coeff=1, error_function=0): #takes a batch, propagates all boards in that batch while accumulating deltaweights. Then sums the deltaweights up and the adjustes the weights of the Network.
        deltaweights_batch = [0]*self.layercount
        selection_size = int(np.round(len(batch.dic)*stoch_coeff))
        if selection_size is 0: #prevent empty selection
            selection_size = 1
        random_selection = random.sample(list(batch.dic.keys()),selection_size)
        for entry in random_selection:
            testdata = self.convert_input(Hashable.unwrap(entry)) #input
            targ = batch.dic[entry].reshape(9*9+1) #target output, this is to be approximated
            if(np.sum(targ)>0): #We can only learn if there are actual target vectors
                targ = targ/np.linalg.norm(targ, ord=1) #normalize (L1-norm)
                y = np.append(testdata,[1]) #We append 1 for the bias
                ys = [0]*self.layercount #y_saved for backpropagation
                
                #Forward-propagate
                for i in range(0,self.layercount): 
                    W = self.weights[i] #anders machen?
                    s = W.dot(y)
                    if i==self.layercount-1: #softmax as activationfct only in last layer    
                        y = np.append(softmax(s),[1]) #We append 1 for the bias
                    else: #in all other hidden layers we use tanh as activation fct
                        if self.activation_function is 0:
                            y = np.append(np.tanh(s),[1]) #We append 1 for the bias
                        else: 
                            if self.activation_function is 1:
                                y = np.append(relu(s),[1]) #We append 1 for the bias
                    ys[i]=y #save the y values for backpropagation
                out=y
                
                #Backpropagation
                
                #Calculate Jacobian of the softmax activationfct in last layer only
                Jacobian_Softmax = [0]*self.layercount
                for i in range(self.layercount-1,self.layercount): #please note that I think this is pure witchcraft happening here
                    yt=ys[i] #load y from ys and lets call it yt y_temporary
                    yt=yt[:-1] #the last entry is from the offset, we don't need this
                    le=len(yt)
                    Jacobian_Softmax_temporary = np.ones((le,le)) #alloc storage temporarily
                    for j in range(0,le):
                        Jacobian_Softmax_temporary[j,:]*=yt[j]
                    Jacobian_Softmax_temporary=np.identity(le) - Jacobian_Softmax_temporary
                    for j in range(0,le):
                        Jacobian_Softmax_temporary[:,j]*=yt[j]
                    Jacobian_Softmax[i]=Jacobian_Softmax_temporary
                #Jacobian_Softmax is quadratic and symmetric.
            
                if self.activation_function is 0:
                    #Calc Jacobian of tanh
                    Jacobian_tanh = [0]*self.layercount
                    for i in range(0,self.layercount): #please note that I think this is pure witchcraft happening here
                        yt=ys[i] #load y from ys and lets call it yt
                        yt=yt[:-1] #the last entry is from the offset, we don't need this
                        u=1-yt*yt
                        Jacobian_tanh[i]=np.diag(u)
                    Jacobian_hidden = Jacobian_tanh
                if self.activation_function is 1:
                    #Calc Jacobian of relu
                    Jacobian_relu = [0]*self.layercount
                    for i in range(0,self.layercount): #please note that I think this is pure witchcraft happening here
                        yt=ys[i] #load y from ys and lets call it yt
                        yt=yt[:-1] #the last entry is from the offset, we don't need this
                        yt[yt>0]=1#actually 0 values go to 1 also. this is not so easy, thus I leave it like that for now
                        Jacobian_relu[i]=np.diag(yt)
                    Jacobian_hidden = Jacobian_relu
                
                #Use (L2) and (L3) to get the error signals of the layers
                errorsignals = [0]*self.layercount
                errorsignals[self.layercount-1] = Jacobian_Softmax[self.layercount-1] # (L2), the error signal of the output layer can be computed directly, here we actually use softmax
                for i in range(2,self.layercount+1):
                    w = self.weights[self.layercount-i+1]
                    DFt = Jacobian_hidden[self.layercount-i] #tanh
                    errdet = np.matmul(w[:,:-1],DFt) #temporary
                    errorsignals[self.layercount-i] = np.dot(errorsignals[self.layercount-i+1],errdet) # (L3), does python fucking get that?
                
                #Use (D3) to compute err_errorsignals as sum over the rows/columns? of the errorsignals weighted by the deriv of the error fct by the output layer. We don't use Lemma 3 dircetly here, we just apply the definition of delta_error
                err_errorsignals=[0]*self.layercount
                if error_function is 0:
                    errorbyyzero = -targ/out[:-1] #Kullback-Leibler divergence derivative
                else:
                    if error_function is 1:
                        errorbyyzero = out[:-1]-targ #Mean-squared-error derivative
                    else:
                        if error_function is 2:
                            errorbyyzero = 1/4*(1-np.sqrt(targ)/np.sqrt(out[:-1])) #Hellinger-distance derivative
                        else:
                            if error_function is 3:
                                errorbyyzero=self.compute_experimental_gradient(out[:-1],targ,1000)
                #errorbyyzero = self.chosen_error_fct(targ,out)
                for i in range(0,self.layercount):
                    err_errorsignals[i]=np.dot(errorbyyzero,errorsignals[i]) #this is the matrix variant of D3
                
                #Use (2.2) to get the sought derivatives. Observe that this is an outer product, though not mentioned in the source (Fuck you Heining, you bastard)
                errorbyweights = [0]*self.layercount #dE/dW
                errorbyweights[0] = np.outer(err_errorsignals[0],testdata).T #Why do I need to transpose here???
                for i in range(1,self.layercount): 
                    errorbyweights[i] = np.outer(err_errorsignals[i-1],ys[i][:-1]) # (L1)
                
                #Compute the change of weights, that means, then apply actualization step of Gradient Descent to weight matrices
                for i in range(0,self.layercount):
                    if type(deltaweights_batch[i]) is int: #initialize
                        deltaweights_batch[i]= -eta*errorbyweights[i]
                    else:
                        deltaweights_batch[i]-=eta*errorbyweights[i]
                    #self.weights[i][:,:-1]= self.weights[i][:,:-1]+ deltaweights[i].T #Problem: atm we only adjust non-bias weights. Change that!
                
                #For Error statistics
                #self.errorsbyepoch.append(self.compute_ms_error (y[:-1], targ))
                #self.abserrorbyepoch.append(self.compute_error (y[:-1], targ))
                #self.KLdbyepoch.append(self.compute_KL_divergence(y[:-1], targ))

        #now adjust weights
        for i in range(0,self.layercount):
            if type(deltaweights_batch[i]) is not int: #in this case we had no target for any board in this batch
                self.weights[i][:,:-1]= self.weights[i][:,:-1]+ deltaweights_batch[i].T #Problem: atm we only adjust non-bias weights. Change that!
        error = self.PropagateSet(batch,error_function)
        return error
    
    def Propagate(self, board):
        #convert board to NeuroNet format (82-dim vector) 
        board = board.vertices
        if len(board) != 82:
            board = board.flatten()
        #Like Heining we are setting: (-1.35:w, 0.45:empty, 1.05:b)
        for i in range(0,len(board)):
            if board[i] == 0:
                board[i] = 0.45
            if board[i] == -1:
                board[i] = -1.35
            if board[i] == 1:
                board[i] = 1.05
                
        y = np.append(board,[1]) #apply offset
        #Forward-propagate
        for i in range(0,self.layercount): 
            W = self.weights[i]
            s = W.dot(y)
            if i==self.layercount-1: #softmax as activationfct only in last layer    
                y = np.append(softmax(s),[1]) #We append 1 for the bias
            elif self.activation_function is 0:
                y = np.append(np.tanh(s),[1]) #We append 1 for the bias
            elif self.activation_function is 1:
                y = np.append(relu(s),[1]) #We append 1 for the bias
        out = y[:-1]
        return out
    
    def PropagateSet(self, testset, error_function=0):
        error = 0
        checked = 0
        for entry in testset.dic:
            testdata = self.convert_input(Hashable.unwrap(entry))
            targ = testset.dic[entry].reshape(9*9+1)
            if(np.sum(targ)>0): #We can only learn if there are actual target vectors
                targ = targ/np.linalg.norm(targ, ord=1) #normalize (L1-norm)
                y = np.append(testdata,[1]) #We append 1 for the bias
                #Forward-propagate
                for i in range(0,self.layercount): 
                    W = self.weights[i]
                    s = W.dot(y)
                    if i==self.layercount-1: #softmax as activationfct only in last layer    
                        y = np.append(softmax(s),[1]) #We append 1 for the bias
                    else: #in all other hidden layers we use tanh as activation fct
                        if self.activation_function is 0:
                            y = np.append(np.tanh(s),[1]) #We append 1 for the bias
                        else: 
                            if self.activation_function is 1:
                                y = np.append(relu(s),[1]) #We append 1 for the bias
                #sum up the error
                if error_function is 0:
                    error += self.compute_KL_divergence(y[:-1], targ)
                else:
                    if error_function is 1:
                        error += self.compute_ms_error(y[:-1], targ)
                    else:
                        if error_function is 2:
                            error += self.compute_Hellinger_dist(y[:-1], targ)
                        else:
                            if error_function is 3:
                                error += self.compute_experimental(y[:-1], targ, 1000)
                checked += 1
        if checked is 0:
            print("The Set contained no feasible boards")
            return
        else:
            error = error/checked #average over training set
            return error
    
    ### Plot results and error:
    def visualize(self, firstout, out, targ):
        games=len(out)
        f, axa = plt.subplots(3, sharex=True)
        axa[0].set_title("Error Plot")
        axa[0].set_ylabel("Mean square Error")
        axa[0].plot(range(0,games),self.errorsbyepoch)
        
        axa[1].set_ylabel("Abs Error")
        axa[1].set_ylim( 0, 2 )
        axa[1].plot(range(0,games),self.abserrorbyepoch)
        
        axa[2].set_xlabel("Epochs")
        axa[2].set_ylabel("K-L divergence")
        axa[2].plot(range(0,games),self.KLdbyepoch)
        
        #Plot the results:
        plt.figure(1)
        f, axarr = plt.subplots(3, sharex=True)
        axarr[0].set_title("Neuronal Net Output Plot: First epoch vs last epoch vs target")
        axarr[0].bar(range(1,(self.n*self.n+2)), firstout) #output of first epoch
        
        axarr[1].set_ylabel("quality in percentage")
        axarr[1].bar(range(1,(self.n*self.n+2)), out[:-1]) #output of last epoch
        
        axarr[2].set_xlabel("Board field number")
        axarr[2].bar(range(1,(self.n*self.n+2)), targ) #target distribution
    
    def visualize_error(self,errors):
        plt.plot(range(0,len(errors)),errors)
    
                
#Tests

            
def test1():
    PN = PolicyNet()
    testset = TrainingDataSgfPass("dgs",'dan_data_10') 
    epochs=2
    error_by_epoch = []
    for i in range(0,epochs):
        error_by_epoch = PN.Learn(testset,epochs)
    plt.plot(range(0,epochs),error_by_epoch)
    
#test1()
    


def test2(): #passen
    PN = PolicyNet()
    testset = TrainingDataSgfPass("dgs",'dan_data_10')
    error=PN.Learn(testset,2)
    print("No batchs: error",error)
   
#test2()
    