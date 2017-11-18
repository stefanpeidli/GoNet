# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:00:05 2017

@author: Stefan

Tags: Policy-net, Neural Network

"""
#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Specifications of the game
n=9 # 9x9 board

# Parameters of the NN

n_hidden_1 = 100 # 1st layer number of neurons
n_hidden_2 = 100 # 2nd layer number of neurons
n_hidden_3 = n*n # 3rd layer number of neurons
num_input = n*n # data input format (board fields)
num_classes = 3 # data input total classes (empty=0, white=1, black=-1)

layers=[n*n,200,100,512,100,n*n] #please leave the first and last equal zu n^2 for now
layercount = len(layers)-1

#Input Test Data n x n
games=100; #random potentially illegal boards as test data
testdata = np.random.uniform(-1.5,1.5,(games,n*n))
testdata[0] = data #load data
testdata = testdata.round()-0.25 # I use this offset for now, because it is convenient an prevents a zero input
for i in range(1,games):
    testdata[i]=testdata[0]
targ=abs(np.random.normal(0,4,n*n)) #create random target
targ=targ/np.linalg.norm(targ, ord=1) #normalize (L1-norm)




# Initialize the weights
# here by a normal distribution
mu=0
sigma=2
weights=[0]*layercount
for i in range(0,layercount):
    weights[i]=np.random.normal(mu, sigma, (layers[i]+1,layers[i+1]))#edit: the +1 in the input dimension is for the bias
#wout=np.random.normal(mu, 0, (n_hidden_3,num_input)) #How to choose these? They dont represent a layer (?)


#Hint:b=a[:-1,:] this is a handy formulation to delete the last column, Thus ~W=W[:-1,1]


errorsbyepoch=[0]*games


###Function Definition yard
  
# activation function
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# the derivative of the activation fct
def softmax_prime(x):#check if this actually works...
    """Compute the jacobian of the softmax fct of x"""
    SM = softmax(x)
    #jac = np.outer(SM, (np.eye(x.size) - np.transpose(SM)) ) WRONG
    jac = SM.dot(np.eye(x.size)) - np.outer(SM, np.transpose(SM) )
    return jac

def compute_error(suggested, target): #compare the prediction with the answer/target
    totalError = 0
    if suggested == target:
        totalError = 0 #We have no Error if the NN did what was expected of it
    else:
        totalError = y[suggested] - y[target]  #The Error could be like here the difference of output-prob between target and suggested move
    return totalError

def compute_ms_error (suggested, target): #Returns the total mean square error (not tested yet)
    diff = np.absolute(suggested - target)
    Error = 0.5*np.inner(diff,diff)
    return Error



###The actual function
    

layers=[n_hidden_1,n_hidden_2,n_hidden_3]

for epoch in range(0,games):
    y = np.append(testdata[epoch],[1])
    ys = [0]*layercount
    #Forward-propagate
    for i in range(0,layercount): 
        w = np.transpose(weights[i]) #anders machen?
        s = w.dot(y)
        y = np.append(softmax(s),[1])
        ys[i]=y #save the y values for backprop (?)
    out=y
    if(epoch==0):
        firstout=y[:-1] #save first epoch output for comparison and visualization
    
    #Calc derivatives/Jacobian of the activationfct in every layer (i dont have a good feeling about this)
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
    
    #Use (L2) and (L3) to get the error signals of the layers
    errorsignals=[0]*layercount
    errorsignals[layercount-1]=DF[layercount-1] # (L2), the error signal of the output layer can be computed directly
    for i in range(2,layercount+1):
        w=weights[layercount-i+1].T
        errdet=np.matmul(w[:,:-1],DF[layercount-i]) #temporary
        errorsignals[layercount-i]=np.dot(errorsignals[layercount-i+1],errdet) # (L3), does python fucking get that?
    
    #Use (D3) to compute err_errorsignals as sum over the rows/columns? of the errorsignals weighted by the deriv of the error fct by the output layer. We don't use Lemma 3 dircetly here, we just apply the definition of delta_error
    err_errorsignals=[0]*layercount
    errorbyyzero = out[:-1]-targ #gives a dim(out) dimensional vector denoting the derivative of the error fct by the output vector
    for i in range(0,layercount):
        err_errorsignals[i]=np.dot(errorbyyzero,errorsignals[i]) #this is the matrix variant of D3
    
    
    #Use (2.2) to get the sought derivatives. Observe that this is an outer product, though not mentioned in the source (Fuck you Heining, you bastard)
    errorbyweights=[0]*layercount
    errorbyweights[0] = np.outer(err_errorsignals[0],testdata[epoch]).T #Why do i need .T here?
    for i in range(1,layercount): 
        errorbyweights[i]=np.outer(err_errorsignals[i-1],ys[i][:-1]) # (L1), do we need to transpose?
    
    #Compute the change of weights, that means, then apply actualization step of Gradient Descent to weight matrices
    eta=0.1 # learning rate
    deltaweights=[0]*layercount
    for i in range(0,layercount):
        deltaweights[i]=-eta*errorbyweights[i]
        weights[i][:-1,:]= weights[i][:-1,:]+ deltaweights[i] #Problem: atm we only adjust non-bias weights. Change that!

    errorsbyepoch[epoch]=compute_ms_error (y[:-1], targ)

#End of Main Loop
        
suggestedmove=np.argmax(y)


#Plot the error:
plt.figure(0)
plt.title("Error Plot")
plt.ylabel("Mean square Error")
plt.xlabel("Epochs")
plt.plot(range(0,games),errorsbyepoch)

#Plot the results:
f, axarr = plt.subplots(3, sharex=True)
plt.figure(1)
axarr[0].set_title("Neuronal Net Output Plot: First epoch vs last epoch vs target")
#axarr[1].set_title("Target")
axarr[1].set_ylabel("quality in percentage")
axarr[2].set_xlabel("Board field number")
#plt.xticks( range(1,(n*n)+1) )  # this looks super ugly if n is higher than 3 because ticks are to dense then.
#plt.ylim( (0,y[suggestedmove]*1.1) )  # set the ylim to ymin, ymax, the 1.1 is an offset for cosmetic reasons
axarr[1].bar(range(1,(n*n+1)), out[:-1])
axarr[0].bar(range(1,(n*n+1)), firstout)
axarr[2].bar(range(1,(n*n+1)), targ)

"""
#Visualization of the output on a board representation:
#plt.figure(2)
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

print("Suggested move: Field number", suggestedmove+1, "with a value of",np.round(100*y[suggestedmove],2),"%.") 

print("We are ",np.round(compute_error(suggestedmove,2)*100,2),"% away from the right solution move.")#test, lets just say that 2 would be the best move for now

