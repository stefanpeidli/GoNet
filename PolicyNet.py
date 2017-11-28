"""
Created on Wed Nov  8 11:00:05 2017
@author: Stefan Peidli
License: MIT
Tags: Policy-net, Neural Network
"""
import numpy as np
import matplotlib.pyplot as plt
from Hashable import Hashable
from TrainingDataFromSgf import TrainingData

class PolicyNet:
    def __init__(self):
        ### Specifications of the game
        self.n=9 # 9x9 board
        
        ### Parameters of the NN
        self.eta = 0.001 # learning rate
        self.layers = [self.n*self.n,100,100,100,self.n*self.n] #please leave the first and last equal zu n^2 for now
        
        ### Initialize the weights
        # here by a normal distribution N(mu,sigma)
        self.layercount = len(self.layers)-1
        mu, sigma = 0, 0.4
        self.weights=[0]*self.layercount #alloc memory
        for i in range(0,self.layercount):
            self.weights[i]=np.random.normal(mu, sigma, (self.layers[i+1],self.layers[i]+1))#edit: the +1 in the input dimension is for the bias
        
        ### Alloc memory for the error statistics
        #Hint:b=a[:-1,:] this is a handy formulation to delete the last column, Thus ~W=W[:-1,1]
        games=2000
        self.errorsbyepoch=[0]*games #mean squared error
        self.abserrorbyepoch=[0]*games #absolute error
        self.KLdbyepoch=[0]*games #Kullback-Leibler divergence
        
    def read_data(self,dictionary):
        aaaaaa=1
        
    def generate_data(self,games):
        ### Input Test Data n x n
        datamanual= False
        if not datamanual:
            #games=2000; #random potentially illegal boards as test data
            testdata = np.random.uniform(-1.5,1.5,(games,self.n*self.n))
            #testdata[0] = data #load data from another script
            testdata = testdata.round()-0.25 # I use this offset for now, because it is convenient an prevents a zero input
            for i in range(1,games):
                testdata[i]=testdata[0]
        else:#load from Converter script
            games=len(gameslist)
            testdata = gameslist
        
        #random target
        targ=abs(np.random.normal(0,40,self.n*self.n)) #create random target
        targ=targ/np.linalg.norm(targ, ord=1) #normalize (L1-norm)
        """
        #manual target, manual statistics
        targ=np.zeros(n*n)
        targ[4]=5
        targ[50]=2
        targ[75]=1
        targ[30]=10
        targ=targ/np.linalg.norm(targ, ord=1) #normalize (L1-norm)
        """
        return [testdata,targ]
        
    ### Function Definition yard
      
    # activation function
    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    """
    # the derivative of the activation fct, dont need this...
    def softmax_prime(x):#check if this actually works...
        #Compute the jacobian of the softmax fct of x
        SM = softmax(x)
        #jac = np.outer(SM, (np.eye(x.size) - np.transpose(SM)) ) WRONG
        jac = SM.dot(np.eye(x.size)) - np.outer(SM, np.transpose(SM) )
        return jac
    """
    def compute_error(self,suggested, target): #compare the prediction with the answer/target, absolute error
        diff = np.absolute(suggested - target)
        Error = np.inner(diff,np.ones(len(target)))
        return Error
    
    def compute_ms_error (self,suggested, target): #Returns the total mean square error
        diff = np.absolute(suggested - target)
        Error = 0.5*np.inner(diff,diff)
        return Error
    
    def compute_KL_divergence (self,suggested,target): #Compute Kullback-Leibler divergence
        diff=suggested/target
        Error = - np.inner(target*np.log(diff),np.ones(len(target)))
        return Error
    
    ### The actual function
    
    def Learnpropagate(self,eta ,trainingdata):
        for entry in trainingdata.dic:
            #bagga=entry[:-1]
            #bagga=bagga[1:]
            #testdata=np.fromstring(bagga, dtype=int, sep=' ')-0.25
            testdata=Hashable.unwrap(entry)
            targ=trainingdata.dic[entry].reshape(9*9)
            if(np.sum(targ)>0): #We can only learn if there are actual target vectors
                y = np.append(testdata,[1])
                ys = [0]*self.layercount
                #Forward-propagate
                for i in range(0,self.layercount): 
                    W = self.weights[i] #anders machen?
                    s = W.dot(y)
                    if i==self.layercount-1: #softmax as activationfct only in last layer    
                        y = np.append(self.softmax(self,s),[1]) #We append 1 for the bias
                    else: #in all other hidden layers we use tanh as activation fct
                        y = np.append(np.tanh(s),[1]) #We append 1 for the bias
                    ys[i]=y #save the y values for backprop (?)
                out=y
                if 'firstout' not in locals():
                    firstout=y[:-1] #save first epoch output for comparison and visualization
                
                #Backpropagation
                
                #Calc derivatives/Jacobian of the softmax activationfct in every layer (i dont have a good feeling about this): Update: I tested this section, it actually works correctly for sure. ToDo: We need not compute this for all layers, only the last one (only layer that uses softmax...)
                DF=[0]*self.layercount
                for i in range(0,self.layercount): #please note that I think this is pure witchcraft happening here
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
                self.errorsbyepoch.append(self.compute_ms_error (y[:-1], targ))
                self.abserrorbyepoch[0]=self.compute_error (y[:-1], targ)
                self.KLdbyepoch[0]=self.compute_KL_divergence(y[:-1], targ)
                
            return [firstout,out]
   
    
    def Propagate(self,board):
        #convert board to NeuroNet format (81-dim vector. TODO: Check for color of NN-training and flip if needed! For now, for convenience, i just subtract 0.25. We should do it similar to Heining by setting: -1.35:w, 0.45:empty, 1.05:b)
        board=board.vertices
        if len(board)!=81:
            board=board.flatten()
        board=board-0.25
        
        y = np.append(board,[1]) #offset
        ys = [0]*self.layercount
        #Forward-propagate
        for i in range(0,self.layercount): 
            W = self.weights[i] #anders machen?
            s = W.dot(y)
            if i==self.layercount-1: #softmax as activationfct only in last layer    
                y = np.append(self.softmax(s),[1]) #We append 1 for the bias
            else: #in all other hidden layers we use tanh as activation fct
                y = np.append(np.tanh(s),[1]) #We append 1 for the bias
            ys[i]=y #save the y values for backprop (?)
        out=y[:-1]
        move=np.argmax(out)
        x=int(move%9)
        y=int(np.floor(move/9))
        move2=[x,y] #check if this is right, i dont think so. The counting is wrong
        return move2
    
    ### Plot results and error:
    def visualize(self, games, firstout, out, targ):
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
        axarr[0].bar(range(1,(self.n*self.n+1)), firstout) #output of first epoch
        
        axarr[1].set_ylabel("quality in percentage")
        axarr[1].bar(range(1,(self.n*self.n+1)), out[:-1]) #output of last epoch
        
        axarr[2].set_xlabel("Board field number")
        axarr[2].bar(range(1,(self.n*self.n+1)), targ) #target distribution
        
        """
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

#Tests
if 'NN' not in locals():
    NN=PolicyNet()
games=1000
#[testdata,targ] = NN.generate_data(games)
eta=0.01
testdata=TrainingData()
testdata.importTrainingData("dgs",1,10) #load from TDFsgf
for i in range(0,1):
    [firstout,out]=NN.Learnpropagate(eta ,testdata)
print(NN.weights)
#NN.visualize(games,firstout,out,targ) #atm only works if games=2000
#zer=np.zeros(81)
#ag=Board(9)
#eg=NN.Propagate(ag)
#print(ag)
#print(eg)


