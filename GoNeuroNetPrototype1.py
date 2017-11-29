"""
Created on Wed Nov  8 11:00:05 2017
@author: Stefan Peidli
License: MIT
Tags: Policy-net, Neural Network
"""
import numpy as np
import matplotlib.pyplot as plt

### Specifications of the game
n=9 # 9x9 board

### Parameters of the NN
num_input = n*n # data input format (board fields)
num_classes = 3 # data input total classes
eta = 0.001 # learning rate
layers = [n*n,100,100,100,n*n] #please leave the first and last equal zu n^2 for now

### Input Test Data n x n
datamanual= False
if not datamanual:
    games=2000; #random potentially illegal boards as test data
    testdata = np.random.uniform(-1.5,1.5,(games,n*n))
    #testdata[0] = data #load data from another script
    testdata = testdata.round()-0.25 # I use this offset for now, because it is convenient an prevents a zero input
    for i in range(1,games):
        testdata[i]=testdata[0]
else:#won't work until we have real data and statistics to it
    games=len(gameslist)
    testdata = gameslist

#random target
targ=abs(np.random.normal(0,40,n*n)) #create random target
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
### Initialize the weights
# here by a normal distribution N(mu,sigma)
layercount = len(layers)-1
mu, sigma = 0, 0.4
weights=[0]*layercount #alloc memory
for i in range(0,layercount):
    weights[i]=np.random.normal(mu, sigma, (layers[i+1],layers[i]+1))#edit: the +1 in the input dimension is for the bias

### Alloc memory for the error statistics
#Hint:b=a[:-1,:] this is a handy formulation to delete the last column, Thus ~W=W[:-1,1]
errorsbyepoch=[0]*games #mean squared error
abserrorbyepoch=[0]*games #absolute error
KLdbyepoch=[0]*games #Kullback-Leibler divergence

### Function Definition yard
  
# activation function
def softmax(x):
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
def compute_error(suggested, target): #compare the prediction with the answer/target, absolute error
    diff = np.absolute(suggested - target)
    Error = np.inner(diff,np.ones(len(target)))
    return Error

def compute_ms_error (suggested, target): #Returns the total mean square error
    diff = np.absolute(suggested - target)
    Error = 0.5*np.inner(diff,diff)
    return Error

def compute_KL_divergence (suggested,target): #Compute Kullback-Leibler divergence
    diff=suggested/target
    Error = - np.inner(target*np.log(diff),np.ones(len(target)))
    return Error

### The actual function

for epoch in range(0,games):
    y = np.append(testdata[epoch],[1])
    ys = [0]*layercount
    #Forward-propagate
    for i in range(0,layercount): 
        W = weights[i] #anders machen?
        s = W.dot(y)
        if i==layercount-1: #softmax as activationfct only in last layer    
            y = np.append(softmax(s),[1]) #We append 1 for the bias
        else: #in all other hidden layers we use tanh as activation fct
            y = np.append(np.tanh(s),[1]) #We append 1 for the bias
        ys[i]=y #save the y values for backprop (?)
    out=y
    if(epoch==0):
        firstout=y[:-1] #save first epoch output for comparison and visualization
    
    #Backpropagation
    
    #Calc derivatives/Jacobian of the softmax activationfct in every layer (i dont have a good feeling about this): Update: I tested this section, it actually works correctly for sure. ToDo: We need not compute this for all layers, only the last one (only layer that uses softmax...)
    DF=[0]*layercount
    for i in range(0,layercount): #please note that I think this is pure witchcraft happening here
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
    DFtan=[0]*layercount
    for i in range(0,layercount): #please note that I think this is pure witchcraft happening here
        yt=ys[i] #load y from ys and lets call it yt
        yt=yt[:-1] #the last entry is from the offset, we don't need this
        le=len(yt)
        u=1-yt*yt
        DFtan[i]=np.diag(u)
    
    #Use (L2) and (L3) to get the error signals of the layers
    errorsignals=[0]*layercount
    errorsignals[layercount-1]=DF[layercount-1] # (L2), the error signal of the output layer can be computed directly, here we actually use softmax
    for i in range(2,layercount+1):
        w=weights[layercount-i+1]
        DFt=DFtan[layercount-i] #tanh
        errdet=np.matmul(w[:,:-1],DFt) #temporary
        errorsignals[layercount-i]=np.dot(errorsignals[layercount-i+1],errdet) # (L3), does python fucking get that?
    
    #Use (D3) to compute err_errorsignals as sum over the rows/columns? of the errorsignals weighted by the deriv of the error fct by the output layer. We don't use Lemma 3 dircetly here, we just apply the definition of delta_error
    err_errorsignals=[0]*layercount
    #errorbyyzero = out[:-1]-targ #Mean-squared-error
    errorbyyzero = -targ/out[:-1] #Kullback-Leibler divergence
    for i in range(0,layercount):
        err_errorsignals[i]=np.dot(errorbyyzero,errorsignals[i]) #this is the matrix variant of D3
    
    #Use (2.2) to get the sought derivatives. Observe that this is an outer product, though not mentioned in the source (Fuck you Heining, you bastard)
    errorbyweights=[0]*layercount #dE/dW
    errorbyweights[0] = np.outer(err_errorsignals[0],testdata[epoch]).T #Why do I need to transpose here???
    for i in range(1,layercount): 
        errorbyweights[i]=np.outer(err_errorsignals[i-1],ys[i][:-1]) # (L1)
    
    #Compute the change of weights, that means, then apply actualization step of Gradient Descent to weight matrices
    deltaweights=[0]*layercount
    for i in range(0,layercount):
        deltaweights[i]=-eta*errorbyweights[i]
        weights[i][:,:-1]= weights[i][:,:-1]+ deltaweights[i].T #Problem: atm we only adjust non-bias weights. Change that!
    
    #For Error statistics
    errorsbyepoch[epoch]=compute_ms_error (y[:-1], targ)
    abserrorbyepoch[epoch]=compute_error (y[:-1], targ)
    KLdbyepoch[epoch]=compute_KL_divergence(y[:-1], targ)

#End of Main Loop

### Plot results and error:

f, axa = plt.subplots(3, sharex=True)
axa[0].set_title("Error Plot")
axa[0].set_ylabel("Mean square Error")
axa[0].plot(range(0,games),errorsbyepoch)

axa[1].set_ylabel("Abs Error")
axa[1].set_ylim( 0, 2 )
axa[1].plot(range(0,games),abserrorbyepoch)

axa[2].set_xlabel("Epochs")
axa[2].set_ylabel("K-L divergence")
axa[2].plot(range(0,games),KLdbyepoch)

#Plot the results:
plt.figure(1)
f, axarr = plt.subplots(3, sharex=True)
axarr[0].set_title("Neuronal Net Output Plot: First epoch vs last epoch vs target")
axarr[0].bar(range(1,(n*n+1)), firstout) #output of first epoch

axarr[1].set_ylabel("quality in percentage")
axarr[1].bar(range(1,(n*n+1)), out[:-1]) #output of last epoch

axarr[2].set_xlabel("Board field number")
axarr[2].bar(range(1,(n*n+1)), targ) #target distribution

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

#



#print("We are ",np.round(compute_error(suggestedmove,2)*100,2),"% away from the right solution move.")#test, lets just say that 2 would be the best move for now

