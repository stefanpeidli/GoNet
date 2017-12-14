# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:06:54 2017

@author: Stefan
"""
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def KLD ( suggested, target): #Compute Kullback-Leibler divergence, now stable!
    t=target[target!=0] #->we'd divide by 0 else, does not have inpact on error anyway ->Problem: We don't punish the NN for predicting non-zero values on zero target!
    s=suggested[target!=0]
    difference=s/t #this is stable
    Error = - np.inner(t*np.log(difference),np.ones(len(t)))
    return Error

def KLD2( suggested, target): #Compute Kullback-Leibler divergence, now stable!
    t=target
    t=t+1e-100 #->we'd divide by 0 else, does not have inpact on error anyway ->Problem: We don't punish the NN for predicting non-zero values on zero target!
    s=suggested
    s=s+1e-100
    difference=s/t #this is stable
    Error = - np.inner(t*np.log(difference),np.ones(len(t)))
    return Error

def KLDGRAD(sug,targ):
    g=np.zeros(len(targ))
    sug=sug+1e-100
    for i in range(0,len(targ)):
        if sug[i]!=0:
            g[i]=-targ[i]/sug[i]
    return g

#error fct Number 1
def MSE ( suggested, target): #Returns the total mean square error
    difference = np.absolute(suggested - target)
    Error = 0.5*np.inner(difference,difference)
    return Error

#error fct Number 2
def HELDIST ( suggested, target):
    return np.linalg.norm(np.sqrt(suggested)-np.sqrt(target), ord=2) /np.sqrt(2)

#error fct Number 3
def CROSSENTRO ( suggested, target):
    return ENTRO(target) + KLD(suggested,target) #wie rum kldiv?  

#error fct Number 4
def EXPE ( suggested, target, gamma):
    alpha = 1/gamma
    beta = np.log(gamma)
    error = alpha*np.sum(np.exp((suggested - target)*beta))
    return error

def EXPEGRAD( suggested, target, gamma=1000):
    alpha = 1/gamma
    beta = np.log(gamma)
    gradient = alpha*beta*np.exp((suggested - target)*beta)
    return gradient
    
#error fct Number x, actually not a good one. Only for statistics
def MAE (suggested, target): #compare the prediction with the answer/target, absolute error
    difference = np.absolute(suggested - target)
    Error = np.inner(difference,np.ones(len(target)))
    return Error

def ENTRO (distribution):
    return -np.inner(distribution[distribution!=0],np.log(distribution[distribution!=0]))


le=5
y1=np.zeros(le)
y1[1]=1

y3=np.zeros(le)
y3[3]=1-1/le
y3[1]=1/le

yunif=np.ones(le)/le

w=1/5
ali='center'





for i in [y3,yunif]:
    print(np.round(i,2))
    print(np.round(y1,2))
    print("KLD",KLD(i,y1))
    print("KLD2",KLD2(i,y1))
    print("MSE",MSE(i,y1))
    print("HELDIST",HELDIST(i,y1))
    print("CROSSENTRO",CROSSENTRO(i,y1))
    print("EXPE",EXPE(i,y1,1000))
    print("EXPE2",EXPE(y1,i,1000))
    print("MAE",MAE(i,y1))
    print(" ")

y0=yunif
eta=0.01

y1=np.array([0,2,4,2,0])


y1=y1/np.sum(y1)

plt.grid(True)
plt.bar(np.arange(le)-w,y1,width=w,align=ali,color='b')
bp = mpatches.Patch(color='blue', label='Target')
plt.bar(np.arange(le),y0,width=w,align=ali,color='r')
rp = mpatches.Patch(color='red', label='Start')
gp = mpatches.Patch(color='green', label='Stop')
plt.legend(handles=[bp,rp,gp])


for j in range(0,800):
    yn=KLDGRAD(y0,y1)
    #yn=y0-y1
    y0=np.abs(y0-eta*yn)
    y0=y0/np.inner(y0,np.ones(le))
    print(KLD(y0,y1))
print(np.round(y1,2))
print(np.round(y0,2))
plt.bar(np.arange(le)+2*w,y0,width=w,align=ali,color='y')
#plt.show()

for i in [y0]:
    print("KLD",KLD(i,y1))
    print("KLD2",KLD2(i,y1))
    print("MSE",MSE(i,y1))
    print("HELDIST",HELDIST(i,y1))
    print("CROSSENTRO",CROSSENTRO(i,y1))
    print("EXPE",EXPE(i,y1,1000))
    print("EXPE2",EXPE(y1,i,1000))
    print("MAE",MAE(i,y1))
    print(" ")

#########
y0=yunif
y=[[0,1,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,1,0]]

for i in range(100):
    random.shuffle(y)
    for j in y:
        y1=j
        yn=KLDGRAD(y0,y1)
        #yn=y0-y1
        y0=np.abs(y0-eta*yn)
        y0=y0/np.inner(y0,np.ones(le))
print(np.round(y1,2))
print(np.round(y0,2))
plt.bar(np.arange(le)+w,y0,width=w,align=ali,color='g')
plt.show()

for i in [y0]:
    print("KLD",KLD(i,y1))
    print("KLD2",KLD2(i,y1))
    print("MSE",MSE(i,y1))
    print("HELDIST",HELDIST(i,y1))
    print("CROSSENTRO",CROSSENTRO(i,y1))
    print("EXPE",EXPE(i,y1,1000))
    print("EXPE2",EXPE(y1,i,1000))
    print("MAE",MAE(i,y1))
    print(" ")