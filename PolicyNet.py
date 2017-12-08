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
from TrainingDataFromSgf import TrainingDataSgf #Better version
from TrainingDataFromSgf import TrainingDataSgfPass #82 not 81
import os
import time
import datetime
import random

def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

class PolicyNet:
    def __init__(self,layers=[9*9,100,200,300,200,100,9*9+1]):
        ### Specifications of the game
        self.n=9 # 9x9 board
        
        ### Parameters of the NN
        self.eta = 0.001 # learning rate
        self.layers = layers #please leave the first and last equal zu n^2 for now
        
        ### Initialize the weights
        # here by a normal distribution N(mu,sigma)
        self.layercount = len(self.layers)-1
        mu=0
        self.weights=[0]*self.layercount #alloc memory
        for i in range(0,self.layercount):
            sigma = 1/np.sqrt(self.layers[i+1]) #vgl Heining #TODO:Check if inputs are well-behaved (approx. normalized)
            self.weights[i]=np.random.normal(mu, sigma, (self.layers[i+1],self.layers[i]+1))#edit: the +1 in the input dimension is for the bias
        
        ### Alloc memory for the error statistics
        #Hint:b=a[:-1,:] this is a handy formulation to delete the last column, Thus ~W=W[:-1,1]
        
        #self.errorsbyepoch=[] #mean squared error
        #self.abserrorbyepoch=[] #absolute error
        #self.KLdbyepoch=[] #Kullback-Leibler divergence
        
        
    ### Function Definition yard
      
    # error functions

    def compute_abs_error(self, suggested, target): #compare the prediction with the answer/target, absolute error
        difference = np.absolute(suggested - target)
        Error = np.inner(difference,np.ones(len(target)))
        return Error
    
    def compute_ms_error (self, suggested, target): #Returns the total mean square error
        difference = np.absolute(suggested - target)
        Error = 0.5*np.inner(difference,difference)
        return Error
    
    def compute_KL_divergence (self, suggested,target): #Compute Kullback-Leibler divergence, now stable!
        t=target[target!=0] #->we'd divide by 0 else, does not have inpact on error anyway ->Problem: We don't punish the NN for predicting non-zero values on zero target!
        s=suggested[target!=0]
        difference=s/t #this is stable
        Error = - np.inner(t*np.log(difference),np.ones(len(t)))
        return Error
    
    def compute_Hellinger_dist (self, suggested, target):
        return np.linalg.norm(np.sqrt(suggested)-np.sqrt(target), ord=2) /np.sqrt(2)
    
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
    
    ### The actual functions
    
    def Learnsplit(self, eta, trainingdata, trainingrate, tolerance, maxepochs, stoch_coeff):
        N = len(trainingdata.dic)
        splitindex = int(round(N*trainingrate))
        trainingset, testset = TrainingData(), TrainingData() #TODO: check if this is fine with TraingDataSgf method
        trainingset.dic = dict(list(trainingdata.dic.items())[:splitindex])
        testset.dic = dict(list(trainingdata.dic.items())[splitindex:])
        
        error = [tolerance+1]
        epochs = 0
        while error[-1:][0] > tolerance and epochs < maxepochs:
            epochs += 1
            self.Learnpropagate(eta, trainingset, stoch_coeff)
            error.append(self.PropagateSet(testset))
        return [error,epochs]
    
    def splitintobatches(self,trainingdata,batchsize): #splits trainingdata into batches of size batchsize
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
        return[k,Batch_sets]

    
    def Learnpropagate(self, eta, trainingdata, stoch_coeff = 1 ):
        counter, error_in_selection= 0, 0
        random_selection = random.sample(list(trainingdata.dic.keys()),int(np.round(len(trainingdata.dic)*stoch_coeff)))
        for entry in random_selection:
            testdata=self.convert_input(Hashable.unwrap(entry))
            targ=trainingdata.dic[entry].reshape(9*9+1)
            if(np.sum(targ)>0): #We can only learn if there are actual target vectors
                targ=targ/np.linalg.norm(targ, ord=1) #normalize (L1-norm)
                y = np.append(testdata,[1])
                ys = [0]*self.layercount
                #Forward-propagate
                for i in range(0,self.layercount): 
                    W = self.weights[i] #anders machen?
                    s = W.dot(y)
                    if i==self.layercount-1: #softmax as activationfct only in last layer    
                        y = np.append(softmax(s),[1]) #We append 1 for the bias
                    else: #in all other hidden layers we use tanh as activation fct
                        y = np.append(np.tanh(s),[1]) #We append 1 for the bias
                    ys[i]=y #save the y values for backprop (?)
                out=y
                if 'firstout' not in locals():
                    firstout=y[:-1] #save first epoch output for comparison and visualization
                
                #Backpropagation
                
                #Calc derivatives/Jacobian of the softmax activationfct in every layer (i dont have a good feeling about this): Update: I tested this section, it actually works correctly for sure. ToDo: We need not compute this for all layers, only the last one (only layer that uses softmax...)
                DF=[0]*self.layercount
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
                    DF[i]=DFt
                #DF is a Jacobian, thus it is quadratic and symmetric
            
                #Calc Jacobian of tanh
                DFtan=[0]*self.layercount
                for i in range(0,self.layercount): #please note that I think this is pure witchcraft happening here
                    yt=ys[i] #load y from ys and lets call it yt
                    yt=yt[:-1] #the last entry is from the offset, we don't need this
                    le=len(yt)
                    u=1-yt*yt
                    DFtan[i]=np.diag(u)
                
                #Use (L2) and (L3) to get the error signals of the layers
                errorsignals=[0]*self.layercount
                errorsignals[self.layercount-1]=DF[self.layercount-1] # (L2), the error signal of the output layer can be computed directly, here we actually use softmax
                for i in range(2,self.layercount+1):
                    """if i==layercount+1:#softmax
                        w=weights[layercount-i+1]
                        errdet=np.matmul(w[:,:-1],DF[layercount-i]) #temporary
                        errorsignals[layercount-i]=np.dot(errorsignals[layercount-i+1],errdet) # (L3), does python fucking get that?
                    else:"""
                    w=self.weights[self.layercount-i+1]
                    DFt=DFtan[self.layercount-i] #tanh
                    errdet=np.matmul(w[:,:-1],DFt) #temporary
                    errorsignals[self.layercount-i]=np.dot(errorsignals[self.layercount-i+1],errdet) # (L3), does python fucking get that?
                
                #Use (D3) to compute err_errorsignals as sum over the rows/columns? of the errorsignals weighted by the deriv of the error fct by the output layer. We don't use Lemma 3 dircetly here, we just apply the definition of delta_error
                err_errorsignals=[0]*self.layercount
                #errorbyyzero = out[:-1]-targ #Mean-squared-error
                errorbyyzero = -targ/out[:-1] #Kullback-Leibler divergence
                for i in range(0,self.layercount):
                    err_errorsignals[i]=np.dot(errorbyyzero,errorsignals[i]) #this is the matrix variant of D3
                
                #Use (2.2) to get the sought derivatives. Observe that this is an outer product, though not mentioned in the source (Fuck you Heining, you bastard)
                errorbyweights=[0]*self.layercount #dE/dW
                errorbyweights[0] = np.outer(err_errorsignals[0],testdata).T #Why do I need to transpose here???
                for i in range(1,self.layercount): 
                    errorbyweights[i]=np.outer(err_errorsignals[i-1],ys[i][:-1]) # (L1)
                
                #Compute the change of weights, that means, then apply actualization step of Gradient Descent to weight matrices
                deltaweights=[0]*self.layercount
                for i in range(0,self.layercount):
                    deltaweights[i]=-eta*errorbyweights[i]
                    self.weights[i][:,:-1]= self.weights[i][:,:-1]+ deltaweights[i].T #Problem: atm we only adjust non-bias weights. Change that!
                
                #For Error statistics
                #self.errorsbyepoch.append(self.compute_ms_error (y[:-1], targ))
                #self.abserrorbyepoch.append(self.compute_error (y[:-1], targ))
                #self.KLdbyepoch.append(self.compute_KL_divergence(y[:-1], targ))
        
        error = self.PropagateSet(trainingdata) #real error
        
        return [firstout,out,error]
    
    def LearnpropagateTest(self, eta, trainingdata, stoch_coeff):
        counter, error_in_selection= 0, 0
        random_selection = random.sample(list(trainingdata.dic.keys()),int(np.round(len(trainingdata.dic)*stoch_coeff)))
        for entry in random_selection:
            testdata=self.convert_input(Hashable.unwrap(entry))
            targ=trainingdata.dic[entry].reshape(9*9+1)
            if(np.sum(targ)>0): #We can only learn if there are actual target vectors
                targ=targ/np.linalg.norm(targ, ord=1) #normalize (L1-norm)
                y = np.append(testdata,[1])
                ys = [0]*self.layercount
                #Forward-propagate
                for i in range(0,self.layercount): 
                    W = self.weights[i] #anders machen?
                    s = W.dot(y)
                    if i==self.layercount-1: #softmax as activationfct only in last layer    
                        y = np.append(softmax(s),[1]) #We append 1 for the bias
                    else: #in all other hidden layers we use tanh as activation fct
                        y = np.append(np.tanh(s),[1]) #We append 1 for the bias
                    ys[i]=y #save the y values for backprop (?)
                out=y
                if 'firstout' not in locals():
                    firstout=y[:-1] #save first epoch output for comparison and visualization
                
                #Backpropagation
                
                #Calc derivatives/Jacobian of the softmax activationfct in every layer (i dont have a good feeling about this): Update: I tested this section, it actually works correctly for sure. ToDo: We need not compute this for all layers, only the last one (only layer that uses softmax...)
                DF=[0]*self.layercount
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
                    DF[i]=DFt
                #DF is a Jacobian, thus it is quadratic and symmetric
            
                #Calc Jacobian of tanh
                DFtan=[0]*self.layercount
                for i in range(0,self.layercount): #please note that I think this is pure witchcraft happening here
                    yt=ys[i] #load y from ys and lets call it yt
                    yt=yt[:-1] #the last entry is from the offset, we don't need this
                    le=len(yt)
                    u=1-yt*yt
                    DFtan[i]=np.diag(u)
                
                #Use (L2) and (L3) to get the error signals of the layers
                errorsignals=[0]*self.layercount
                errorsignals[self.layercount-1]=DF[self.layercount-1] # (L2), the error signal of the output layer can be computed directly, here we actually use softmax
                for i in range(2,self.layercount+1):
                    w=self.weights[self.layercount-i+1]
                    DFt=DFtan[self.layercount-i] #tanh
                    errdet=np.matmul(w[:,:-1],DFt) #temporary
                    errorsignals[self.layercount-i]=np.dot(errorsignals[self.layercount-i+1],errdet) # (L3), does python fucking get that?
                
                #Use (D3) to compute err_errorsignals as sum over the rows/columns? of the errorsignals weighted by the deriv of the error fct by the output layer. We don't use Lemma 3 dircetly here, we just apply the definition of delta_error
                err_errorsignals=[0]*self.layercount
                #errorbyyzero = out[:-1]-targ #Mean-squared-error
                errorbyyzero = 1/4*(1-np.sqrt(targ)/np.sqrt(out[:-1])) #Hellinger-distance
                for i in range(0,self.layercount):
                    err_errorsignals[i]=np.dot(errorbyyzero,errorsignals[i]) #this is the matrix variant of D3
                
                #Use (2.2) to get the sought derivatives. Observe that this is an outer product, though not mentioned in the source (Fuck you Heining, you bastard)
                errorbyweights=[0]*self.layercount #dE/dW
                errorbyweights[0] = np.outer(err_errorsignals[0],testdata).T #Why do I need to transpose here???
                for i in range(1,self.layercount): 
                    errorbyweights[i]=np.outer(err_errorsignals[i-1],ys[i][:-1]) # (L1)
                
                #Compute the change of weights, that means, then apply actualization step of Gradient Descent to weight matrices
                deltaweights=[0]*self.layercount
                for i in range(0,self.layercount):
                    deltaweights[i]=-eta*errorbyweights[i]
                    self.weights[i][:,:-1]= self.weights[i][:,:-1]+ deltaweights[i].T #Problem: atm we only adjust non-bias weights. Change that!
                
                #For Error statistics
                #self.errorsbyepoch.append(self.compute_ms_error (y[:-1], targ))
                #self.abserrorbyepoch.append(self.compute_error (y[:-1], targ))
                #self.KLdbyepoch.append(self.compute_KL_divergence(y[:-1], targ))
        
        #check error
        error = 0
        checked = 0
        for entry in trainingdata.dic:
            testdata=self.convert_input(Hashable.unwrap(entry))
            targ=trainingdata.dic[entry].reshape(9*9+1)
            if(np.sum(targ)>0): #We can only learn if there are actual target vectors
                targ=targ/np.linalg.norm(targ, ord=1) #normalize (L1-norm)
                y = np.append(testdata,[1])
                ys = [0]*self.layercount
                #Forward-propagate
                for i in range(0,self.layercount): 
                    W = self.weights[i] #anders machen?
                    s = W.dot(y)
                    if i==self.layercount-1: #softmax as activationfct only in last layer    
                        y = np.append(softmax(s),[1]) #We append 1 for the bias
                    else: #in all other hidden layers we use tanh as activation fct
                        y = np.append(np.tanh(s),[1]) #We append 1 for the bias
                    ys[i]=y #save the y values for backprop (?)
                error += self.compute_Hellinger_dist(y[:-1], targ)
                checked += 1
        error = error/checked #average over training set    
        return [firstout,out,error]
    
    def Learnpropagatebatch(self, eta, batch, stoch_coeff): #takes a batch, propagates all boards in that batch while accumulating deltaweights. Then sums the deltaweights up and the adjustes the weights of the Network.
        deltaweights_batch=[0]*self.layercount
        selection_size = int(np.round(len(batch.dic)*stoch_coeff))
        if selection_size is 0: #prevent empty selection
            selection_size = 1
        random_selection = random.sample(list(batch.dic.keys()),selection_size)
        for entry in random_selection:
            testdata=self.convert_input(Hashable.unwrap(entry))
            targ=batch.dic[entry].reshape(9*9+1)
            if(np.sum(targ)>0): #We can only learn if there are actual target vectors
                targ=targ/np.linalg.norm(targ, ord=1) #normalize (L1-norm)
                y = np.append(testdata,[1])
                ys = [0]*self.layercount
                #Forward-propagate
                for i in range(0,self.layercount): 
                    W = self.weights[i] #anders machen?
                    s = W.dot(y)
                    if i==self.layercount-1: #softmax as activationfct only in last layer    
                        y = np.append(softmax(s),[1]) #We append 1 for the bias
                    else: #in all other hidden layers we use tanh as activation fct
                        y = np.append(np.tanh(s),[1]) #We append 1 for the bias
                    ys[i]=y #save the y values for backprop (?)
                out=y
                if 'firstout' not in locals():
                    firstout=y[:-1] #save first epoch output for comparison and visualization
                
                #Backpropagation
                
                #Calc derivatives/Jacobian of the softmax activationfct in every layer (i dont have a good feeling about this): Update: I tested this section, it actually works correctly for sure. ToDo: We need not compute this for all layers, only the last one (only layer that uses softmax...)
                DF=[0]*self.layercount
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
                    DF[i]=DFt
                #DF is a Jacobian, thus it is quadratic and symmetric
            
                #Calc Jacobian of tanh
                DFtan=[0]*self.layercount
                for i in range(0,self.layercount): #please note that I think this is pure witchcraft happening here
                    yt=ys[i] #load y from ys and lets call it yt
                    yt=yt[:-1] #the last entry is from the offset, we don't need this
                    le=len(yt)
                    u=1-yt*yt
                    DFtan[i]=np.diag(u)
                
                #Use (L2) and (L3) to get the error signals of the layers
                errorsignals=[0]*self.layercount
                errorsignals[self.layercount-1]=DF[self.layercount-1] # (L2), the error signal of the output layer can be computed directly, here we actually use softmax
                for i in range(2,self.layercount+1):
                    """if i==layercount+1:#softmax
                        w=weights[layercount-i+1]
                        errdet=np.matmul(w[:,:-1],DF[layercount-i]) #temporary
                        errorsignals[layercount-i]=np.dot(errorsignals[layercount-i+1],errdet) # (L3), does python fucking get that?
                    else:"""
                    w=self.weights[self.layercount-i+1]
                    DFt=DFtan[self.layercount-i] #tanh
                    errdet=np.matmul(w[:,:-1],DFt) #temporary
                    errorsignals[self.layercount-i]=np.dot(errorsignals[self.layercount-i+1],errdet) # (L3), does python fucking get that?
                
                #Use (D3) to compute err_errorsignals as sum over the rows/columns? of the errorsignals weighted by the deriv of the error fct by the output layer. We don't use Lemma 3 dircetly here, we just apply the definition of delta_error
                err_errorsignals=[0]*self.layercount
                #errorbyyzero = out[:-1]-targ #Mean-squared-error
                errorbyyzero = -targ/out[:-1] #Kullback-Leibler divergence
                for i in range(0,self.layercount):
                    err_errorsignals[i]=np.dot(errorbyyzero,errorsignals[i]) #this is the matrix variant of D3
                
                #Use (2.2) to get the sought derivatives. Observe that this is an outer product, though not mentioned in the source (Fuck you Heining, you bastard)
                errorbyweights=[0]*self.layercount #dE/dW
                errorbyweights[0] = np.outer(err_errorsignals[0],testdata).T #Why do I need to transpose here???
                for i in range(1,self.layercount): 
                    errorbyweights[i]=np.outer(err_errorsignals[i-1],ys[i][:-1]) # (L1)
                
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
        error = self.PropagateSet(batch)
        return [firstout,out,error]
    
    def Propagate(self,board):
        #convert board to NeuroNet format (81-dim vector) 
        #TODO: Check for color of NN-training and flip if needed! For now, for convenience, i just subtract 0.25. We should do it similar to Heining by setting: -1.35:w, 0.45:empty, 1.05:b)
        board=board.vertices
        if len(board)!=81:
            board=board.flatten()
        for i in range(0,len(board)):
            if board[i]==0:
                board[i] = 0.45
            if board[i]==-1:
                board[i] = -1.35
            if board[i]==1:
                board[i] = 1.05

        
        y = np.append(board,[1]) #offset
        ys = [0]*self.layercount
        #Forward-propagate
        for i in range(0,self.layercount): 
            W = self.weights[i] #anders machen?
            s = W.dot(y)
            if i==self.layercount-1: #softmax as activationfct only in last layer    
                y = np.append(softmax(s),[1]) #We append 1 for the bias
            else: #in all other hidden layers we use tanh as activation fct
                y = np.append(np.tanh(s),[1]) #We append 1 for the bias
            ys[i]=y #save the y values for backprop (?)
        out=y[:-1]
        return out
    
    def PropagateSet(self,testset):
        error = 0
        checked = 0
        for entry in testset.dic:
            testdata=self.convert_input(Hashable.unwrap(entry))
            targ=testset.dic[entry].reshape(9*9+1)
            if(np.sum(targ)>0): #We can only learn if there are actual target vectors
                targ=targ/np.linalg.norm(targ, ord=1) #normalize (L1-norm)
                y = np.append(testdata,[1])
                #Forward-propagate
                for i in range(0,self.layercount): 
                    W = self.weights[i] #anders machen?
                    s = W.dot(y)
                    if i==self.layercount-1: #softmax as activationfct only in last layer    
                        y = np.append(softmax(s),[1]) #We append 1 for the bias
                    else: #in all other hidden layers we use tanh as activation fct
                        y = np.append(np.tanh(s),[1]) #We append 1 for the bias
                error += self.compute_KL_divergence(y[:-1], targ)
                checked += 1
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
    
        """
    def visualize_board(self):
        
        #Visualization of the output on a board representation:
        #plt.figure(3)
        image = np.zeros(n*n)
        image = out[:-1]
        image = image.reshape((n, n)) # Reshape things into a nxn grid.
        row_labels = reversed(np.array(range(n))+1) #fuck Python
        col_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        plt.matshow(image)
        plt.xticks(range(n), col_labels)
        plt.yticks(range(n), row_labels)
        plt.show()
        """
        
        
        """ To be added (?)
        #Visualization of the input on a board representation:
        image = np.zeros(n*n)
        image = testdata[0]
        image = image.reshape((n, n)) # Reshape things into a nxn grid.
        row_labels = reversed(np.array(range(n))+1) #fuck Python
        col_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        plt.matshow(image)
        plt.xticks(range(n), col_labels)
        plt.yticks(range(n), row_labels)
        plt.show()
        """
    
    #print("We are ",np.round(compute_error(suggestedmove,2)*100,2),"% away from the right solution move.")#test, lets just say that 2 would be the best move for now

    def saveweights(self,folder,filename):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = dir_path + "/" + folder + "/" + filename
        np.savez(file,self.weights)
        
    def loadweightsfromfile(self,folder,filename):
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
                
#Tests

def test(): #passing
    testset1 = TrainingDataSgf("dgs",range(0,5))
    testset2 = TrainingDataSgfPass("dgs",range(0,5))
    for entry in testset1.dic:
        data=Hashable.unwrap(entry)
        targ=testset1.dic[entry].reshape(9*9+1)
        print(len(targ))
    for entry in testset2.dic:
        data=Hashable.unwrap(entry)
        targ=testset2.dic[entry].reshape(9*9+1)
        print(len(targ))
#test()
        
def test2():#bug in error display?
    PN = PolicyNet()
    testset = TrainingDataSgfPass("dgs",range(0,5)) #one game
    error1 = PN.PropagateSet(testset)
    print(error1)
    
    [f,o,error2] = PN.Learnpropagate(0.001,testset,0.8)
    print(error2)
    
    [f,o,error3] = PN.Learnpropagatebatch(0.001,testset,0.8)
    print(error3)
    
#test2()

def test3():
    PN = PolicyNet()
    testset = TrainingDataSgfPass("dgs",range(0,5)) #one game
    for entry in testset.dic:
            testdata=PN.convert_input(Hashable.unwrap(entry))

#test3()
            
def test4():
    PN = PolicyNet()
    testset = TrainingDataSgfPass("dgs",'dan_data_10') 
    epochs=2
    error_by_epoch = []
    for i in range(0,epochs):
        [f,o,e] = PN.LearnpropagateTest(0.01,testset,0.8)
        error_by_epoch.append(e)
    plt.plot(range(0,epochs),error_by_epoch)
    
#test4()
    
def test5(): #bug
    PN = PolicyNet()
    testset = TrainingDataSgfPass("dgs",'dan_data_10')
    t=time.time()
    [f,o,e1]=PN.Learnpropagate(0.01,testset,0.8)
    print("No batchs: error",e1,"time",time.time()-t)
    
    [k,batch]=PN.splitintobatches(testset,60)

    e2=[]
    t=time.time()
    for i in range (0,k):
        [f,o,et2]=PN.Learnpropagatebatch(0.01,batch[i],0.8)
        e2.append(et2)
    e2=np.sum(e2)/len(e2)
    print("Batch size 60: error",e2,"time",time.time()-t)
    
#test5()

def test6(): #passen
    PN = PolicyNet()
    testset = TrainingDataSgfPass("dgs",'dan_data_10')
    [f,o,e1]=PN.Learnpropagate(0.01,testset,0.8)
    print("No batchs: error",e1)
   
#test6()
    