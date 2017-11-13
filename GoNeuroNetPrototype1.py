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
n=9 # 3x3 board

# Parameters of the NN
layercount = 3
n_hidden_1 = 10 # 1st layer number of neurons
n_hidden_2 = 40 # 2nd layer number of neurons
n_hidden_3 = n*n # 3rd layer number of neurons
num_input = n*n # data input format (board fields)
num_classes = 3 # data input total classes (empty=0, white=1, black=-1)


#Input Test Data n x n
games=4; #random potentially illegal boards as test data
testdata = np.random.uniform(-1.5,1.5,(games,n*n))
testdata = testdata.round()
targ=abs(np.random.normal(0,4,n*n)) #create random target
targ=targ/np.linalg.norm(targ, ord=1) #normalize (L1-norm)




# Initialize the weights
# here by a normal distribution
mu=0
sigma=1

w1=np.random.normal(mu, sigma, (num_input,n_hidden_1))
w2=np.random.normal(mu, sigma, (n_hidden_1,n_hidden_2))
w3=np.random.normal(mu, sigma, (n_hidden_2,n_hidden_3))
wout=np.random.normal(mu, sigma, (n_hidden_3,num_input))

weights=[w1,w2,w3]



#biases = { # do we need bias?
    #'b1': tf.Variable(tf.random_normal([n_hidden_1])),
 #   'b2': tf.Variable(tf.random_normal([n_hidden_2])),
  #  'out': tf.Variable(tf.random_normal([num_classes]))
#}


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

for j in range(0,games):
    y = testdata[j]
    ys = [0]*layercount
    #Forward-propagate
    for i in range(0,layercount): 
        w = np.transpose(weights[i]) #anders machen!
        s = w.dot(y)
        y = softmax(s)
        ys[i]=y #save the y values for backprop (?)
    out=y
    
    #Calc derivatives/Jacobian of the activationfct in every layer (i dont have a good feeling about this)
    DF=[1,2,3]
    for i in range(0,layercount): #please note that I think this is pure witchcraft happening here
        yt=ys[i] #load y from ys and lets call it yt
        le=len(yt)
        print(le)
        DFt=np.ones((le,le)) #alloc storage temporarily
        for j in range(0,le):
            DFt[j,:]*=yt[j]
        DFt=np.identity(le) - DFt
        for j in range(0,le):
            DFt[:,j]*=yt[j]
        DF[i]=DFt
    #DF is a Jacobian, thus is quadratic and symmetric
    
    #Use (L2) and (L3) to get the error signals of the layers
    errorsignals=[1,2,3]
    errorsignals[layercount-1]=DF[layercount-1] # (L2), the error signal of the output layer can be computed directly
    for i in range(2,layercount+1):
        errdet=np.dot(weights[layercount-i+1].T,DF[layercount-i]) #temporary
        errorsignals[layercount-i]=np.dot(errorsignals[layercount-i+1].T,errdet) # (L3), does python fucking get that?
    
    #Use (L1) to get the sought derivatives
    Dy=[1,2,3]
    for i in range(0,layercount): #this does somehow not work, dimensions dont fit
        Dy[i]=np.matmul(errorsignals[i],ys[layercount-1-i]) # (L1), do we need to transpose?
    
    print(Dy[0].shape,Dy[1].shape)
    
    #Compute the Gradient of the error fct for Gradient descent
    err=[1,2,3]
    for i in range(0,layercount):
        diff = y - targ
        wdiff = diff * Dy[i]
        err[i]=wdiff
        
    #Apply Gradient Descent to weight matrices
    eta=0.1 # learning rate
    for i in [1]: #range(0,layercount):
        weights[i]-=eta*err[i]
        
    

#End of Main Loop
        
suggestedmove=np.argmax(y)

a=np.ones((3,3))
a[:,0]*=3
a[0,:]+=1
#print(a)
#print(np.dot(a,a))

#Plot the results:
plt.title("Neuronal Net Output Plot")
plt.ylabel("quality in percentage")
plt.xlabel("Board field number")
#plt.xticks( range(1,(n*n)+1) )  # this looks super ugly if n is higher than 3 because ticks are to dense then.
plt.ylim( (0,y[suggestedmove]*1.1) )  # set the ylim to ymin, ymax, the 1.1 is an offset for cosmetic reasons
plt.bar(range(1,(n*n+1)), y)

#Visualization of the output on a board representation:
image = np.zeros(n*n)
image = y
image = image.reshape((n, n)) # Reshape things into a nxn grid.
row_labels = reversed(np.array(range(n))+1) #fuck Python
col_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
plt.matshow(image)
plt.xticks(range(n), col_labels)
plt.yticks(range(n), row_labels)
plt.show()



print("Suggested move: Field number", suggestedmove+1, "with a value of",np.round(100*y[suggestedmove],2),"%.") 

  

#Learning Part (highly experimental!!!)



print("We are ",np.round(compute_error(suggestedmove,2)*100,2),"% away from the right solution move.")#test, lets just say that 2 would be the best move for now

"""
def step_gradient(weights, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]
    
"""



