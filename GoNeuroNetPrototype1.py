# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:00:05 2017

@author: Stefan
"""
#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Specifications of the game
n=3 # 3x3 board

# Parameters of the NN
layercount = 2
n_hidden_1 = 10 # 1st layer number of neurons
n_hidden_2 = 9 # 2nd layer number of neurons
num_input = n*n # data input format (board fields)
num_classes = 3 # data input total classes (empty=0, white=1, black=-1)


#Input Test Data n x n
games=10; #10 empty games as test data
testdata = np.random.uniform(-1.5,1.5,(games,n*n))
testdata = testdata.round()


# Initialize the weights
# here by a normal distribution
mu=0
sigma=1

w1=np.random.normal(mu, sigma, (num_input,n_hidden_1))
w2=np.random.normal(mu, sigma, (n_hidden_1,n_hidden_2))
wout=np.random.normal(mu, sigma, (n_hidden_2,num_input))

weights=[w1,w2,wout]



#biases = { # do we need bias?
    #'b1': tf.Variable(tf.random_normal([n_hidden_1])),
 #   'b2': tf.Variable(tf.random_normal([n_hidden_2])),
  #  'out': tf.Variable(tf.random_normal([num_classes]))
#}



  
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


#The actual function
layers=[n_hidden_1,n_hidden_2]

for j in range(0,games):
    y = testdata[j]
    for i in range(0,layercount):
        w = np.transpose(weights[i]) #anders machen!
        s = w.dot(y)
        y = softmax(s)
    #print(y)
    
suggestedmove=np.argmax(y)

plt.ylabel("quality in percentage")
plt.xticks(range(1,10))
plt.ylim( (0,y[suggestedmove]*1.3) )  # set the ylim to ymin, ymax
plt.bar(range(1,10), y)



print("Suggested move: Field number", suggestedmove+1, "with a value of",np.round(100*y[suggestedmove],2),"%.") 

  

#Learning Part (highly experimental!!!)
def compute_error(suggested, target): #compare the prediction with the answer/target
    totalError = 0
    if suggested == target:
        totalError = 0 #We have no Error if the NN did what was expected of it
    else:
        totalError = y[suggested] - y[target]  #The Error could be like here the difference of output-prob between target and suggested move
    return totalError

print("We are ",np.round(compute_error(suggestedmove,2)*100,2),"% away from the right solution move.")#test

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

#Backpropagation
#print(softmax_prime(y))





