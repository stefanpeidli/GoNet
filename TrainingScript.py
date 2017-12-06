# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:58:03 2017

@author: Stefan Peidli
"""
import numpy as np
import matplotlib.pyplot as plt
from Hashable import Hashable
from TrainingDataFromSgf import TrainingData #legacy
from TrainingDataFromSgf import TrainingDataSgf
import os
import PolicyNet
import time
import datetime

"""
This Script should be used to train Policy Networks.

Parameters:
    learningrate (aka eta) : The "speed" of learning. Denotes how far we go into the direction of the gradient.
    epochs : Number of epochs of learning. An epoch is finished, when all boards are forward and backward propagated once.
    sgfrange : when using all games in dgs, sgfrange denotes the number of game indices that should be added to the training data
"""

def TrainingBasic(PolicyNetwork,learningrate,epochs,sgfrange):
    eta = learningrate
    testdata = TrainingDataSgf("dgs",range(0,sgfrange))
    errors_by_epoch=[]
    for i in range(0,epochs):
        [f,o,error] = PolicyNetwork.Learnpropagate(eta ,testdata)
        errors_by_epoch.append(error)
    return errors_by_epoch

    
def TrainingSplit(PolicyNetwork,larningrate):
    NN = PolicyNet()
    eta = 0.05
    trainingdata = TrainingDataSgf("dgs",range(0,1000))
    datasize = len(trainingdata.dic)
    trainingrate = 0.9
    tolerance = 0.8
    maxepochs = 200
    [error,epochs]=NN.Learnsplit(eta,trainingdata, trainingrate, tolerance, maxepochs)
    print("Datasize was",datasize,",K-L-Error:",error[-1:][0],",Epochs:",epochs)
    
#test2()
    
def test3():
    TestNet = PolicyNet()
    testset = TrainingDataSgf("dgs","dan_data_10")
    epochs=20
    eta = 0.01
    t=time.time()
    initerror = TestNet.PropagateSet(testset)
    print("Propagation took",np.round(time.time()-t,3),"seconds.")
    print("Learning was done with batch size 1, vanilla Gradient Descent and Learning rate",eta)
    er=[]
    totaltime=time.time()
    for i in range(epochs):
        t=time.time()
        [firstout,out,err]=TestNet.Learnpropagate(eta, testset)
        print("epoch number",i,"took",np.round(time.time()-t,3),"seconds.")
        er.append(err)
    error = TestNet.PropagateSet(testset)
    print(error,'Final Error:')
    print(initerror,'Initial Error:')
    print("total time needed:",time.time()-totaltime)
    
    #save results:
    name="weights"+datetime.datetime.now().strftime("%y%m%d%H%M")+"eta10000"+str(int(eta*10000))+"epochs"+str(epochs)+"batchsize"+"1"+"Dan10"
    TestNet.saveweights('Saved_Weights',name)
    #TestNet.visualize_error(er)

#test3()
   
def test5():
    TestNet = PolicyNet() 
    w=TestNet.weights
    testset = TrainingDataSgf("dgs","dan_data_10")
    t1=time.time()
    errorinit = TestNet.PropagateSet(testset)
    tinit=time.time()-t1
    print('Initial Error:',errorinit,'Time:',tinit)
    epochs=2
    eta=0.001
    
    #Batch learning
    er=[] #errors in all the batches
    batcherror=[] #average batch error in each epoch
    batchsize=700
    t1=time.time()
    [k,batches]=TestNet.splitintobatches(testset,batchsize)
    t2=time.time()
    for epoch in range(epochs):
        for i in range(k): #Learn all batches
            [firstout,out,err]=TestNet.Learnpropagatebatch(eta,batches[i])
            er.append(err)
        batcherror.append(np.sum(er)/len(er))
    t3=time.time()
    
    error = TestNet.PropagateSet(testset)
    print('Final Error:',error,"Time:",t3-t2,"where the splitting took additionally",t2-t1)
    #TestNet.visualize_error(er)
    
    #Now without batchs
    TestNet.weights=w #reset weights
    errors2=[]
    t1=time.time()
    for epoch in range(epochs):
        [firstout,out,err]=TestNet.Learnpropagate(eta,testset)
        errors2.append(err)
    t2=time.time()
    error = TestNet.PropagateSet(testset)
    print('Final Error:',error,"Time:",t2-t1)
        
    #plot results
    #plt.plot(range(epochs),batcherror,'b')
    #plt.plot(range(epochs),errors2,'r')
    
#test5()

