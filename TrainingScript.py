# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:58:03 2017
@author: Stefan Peidli
"""
import numpy as np
import matplotlib.pyplot as plt
from Hashable import Hashable
from TrainingDataFromSgf import TrainingDataSgf
from PolicyNet import PolicyNet
import time
import datetime

"""
This Script should be used to train Policy Networks.

Parameters: 
    I gave an advice for choosing reasonable values. Exceed those limits at your own risk. [minimum, recommendation, maximum]
    
    learningrate (aka eta) : The "speed" of learning. Denotes how far we go into the direction of the gradient. [0, 0.01, 0.5]
    epochs : Number of epochs of learning. An epoch is finished, when all boards are forward and backward propagated once. [1, 3-10-30-100, 1000] On my PC with Dan_10 games (~4200 distinct board entries) 1 epoch took around 25 seconds, keep that in mind.
    sgfrange : when using all games in dgs, sgfrange denotes the number of game indices that should be added to the training data. [1, 300, 1 000 000] importing with range 10000 took 23 seconds. Propagating this might even take hours. Please use around 300 for a reasonable computation time and around multiple thousands of boards.
    batchsize : Data will be split into batchs of this size. Those will be propagatet as whole, then gradient descent is applied. If there is no option to choose batchsize, default is 1. [1, 100, size of board entries] requires further testing.
    trainingrate : Data will be split into a training set and test set by rate [trainingrate : (1-trainingrate)]. Training will be done on the trainingset, then the performance (error) on the testset is checked. [0, 0.8, 1] should heuristically be above 0.6
    tolerance : Error tolerance. Training will stop if error falls below it. [up to now, the error of 4 will not be reached that fast..., take what you like here between 0 and 3]
    maxepochs : Maximal amount of epochs to iterate. Training will stop once this epoch is reached. [1, 30, inf] keep the computation time in mind! in case, better run many small Trainings so the results are not lost!

On saving and loading weights:
    FYI, weights can be loaded and saved into a folder < Saved_Weights > which is already in git.ignore. Please create this folder if it does not exist yet.
    Let's say your trained your PolicyNet called Martha, then:
        Saving weights: Martha.saveweights('Saved_Weights', name)
        Loading weights: Martha.loadweightsfromfile('Saved_Weights',name)
    I also added a proposal for a weight naming format:
        name="weights"+datetime.datetime.now().strftime("%y%m%d%H%M")+"eta10000"+str(int(eta*10000))+"epochs"+str(epochs)+"batchsize"+"1"+"Dan10"
    I think this format is rather self-explaining. E.g.
        weights1712061818eta10000100epochs20batchsize1Dan10
        weights from 2017-12-06 at 18:18 with eta*10 000 = 100, epochs trained = 20, batchsize = 1, Dataset was Dan_10
        
        
"""

def TrainingBasic(PolicyNetwork,learningrate,epochs,sgfrange):
    eta = learningrate
    testdata = TrainingDataSgf("dgs",range(0,sgfrange))
    errors_by_epoch=[]
    for i in range(0,epochs):
        [f,o,error] = PolicyNetwork.Learnpropagate(eta ,testdata)
        errors_by_epoch.append(error)
    #return errors_by_epoch

    
def TrainingSplit(PolicyNetwork, learningrate, maxepochs, sgfrange, trainingrate, tolerance):
    eta = learningrate
    trainingdata = TrainingDataSgf("dgs",range(0,1000))
    datasize = len(trainingdata.dic)
    [error,epochs]=PolicyNetwork.Learnsplit(eta,trainingdata, trainingrate, tolerance, maxepochs)
    print("Datasize was",datasize,",K-L-Error:",error[-1:][0],",Epochs:",epochs)
    
    
def TrainingBasicDan10(PolicyNetwork,learningrate,epochs):
    testset = TrainingDataSgf("dgs","dan_data_10")
    eta = learningrate
    t=time.time()
    initerror = PolicyNetwork.PropagateSet(testset)
    print("Propagation took",np.round(time.time()-t,3),"seconds.")
    print("Learning was done with batch size 1, vanilla Gradient Descent and Learning rate",eta)
    er=[]
    totaltime=time.time()
    for i in range(epochs):
        t=time.time()
        [firstout,out,err]=PolicyNetwork.Learnpropagate(eta, testset)
        #print("epoch number",i,"took",np.round(time.time()-t,3),"seconds.")
        er.append(err)
    error = PolicyNetwork.PropagateSet(testset)
    print(error,'Final Error:')
    print(initerror,'Initial Error:')
    print("total time needed:",time.time()-totaltime)
    
    #save results:
    name="weights"+datetime.datetime.now().strftime("%y%m%d%H%M")+"eta10000"+str(int(eta*10000))+"epochs"+str(epochs)+"batchsize"+"1"+"Dan10"
    PolicyNetwork.saveweights('Saved_Weights',name)

   
def ComparisonTraining1(PolicyNetwork,learningrate,epochs,batchsize):
    w=PolicyNetwork.weights
    testset = TrainingDataSgf("dgs","dan_data_10")
    t1=time.time()
    errorinit = PolicyNetwork.PropagateSet(testset)
    tinit=time.time()-t1
    print('Initial Error:',errorinit,'Time:',tinit)
    eta=learningrate
    
    #Batch learning
    er=[] #errors in all the batches
    batcherror=[] #average batch error in each epoch
    t1=time.time()
    [k,batches]=PolicyNetwork.splitintobatches(testset,batchsize)
    t2=time.time()
    for epoch in range(epochs):
        for i in range(k): #Learn all batches
            [firstout,out,err]=PolicyNetwork.Learnpropagatebatch(eta,batches[i])
            er.append(err)
        batcherror.append(np.sum(er)/len(er))
    t3=time.time()
    
    error = PolicyNetwork.PropagateSet(testset)
    print('Final Error:',error,"Time:",t3-t2,"where the splitting took additionally",t2-t1)
    
    #Now without batchs
    PolicyNetwork.weights=w #reset weights
    errors2=[]
    t1=time.time()
    for epoch in range(epochs):
        [firstout,out,err]=PolicyNetwork.Learnpropagate(eta,testset)
        errors2.append(err)
    t2=time.time()
    error = PolicyNetwork.PropagateSet(testset)
    print('Final Error:',error,"Time:",t2-t1)        
    #plot results
    #plt.plot(range(epochs),batcherror,'b')
    #plt.plot(range(epochs),errors2,'r')
    

# Training Area = The Neural Network Gym : Do training here
    
your_name="Faruk"

# example for training:
if your_name is "Example":
    MyNetwork = PolicyNet()
    learningrate = 0.01
    epochs = 5 #one epoch ~ 25 seconds
    sgfrange = 10
    TrainingBasic(MyNetwork , learningrate, epochs, sgfrange)
    name="weights"+datetime.datetime.now().strftime("%y%m%d%H%M")+"eta10000"+str(int(learningrate*10000))+"epochs"+str(epochs)+"batchsize"+"1"+"sgfrange"+str(sgfrange)
    MyNetwork.saveweights('Saved_Weights', name)

# Stefan:
if your_name is "Stefan":
    #hier schreibe ich mein training rein
    print("halo I bims")
    

# Faruk
if your_name is "Faruk":
    MyNetwork = PolicyNet()  
    #w=MyNetwork.weights
    learningrate = 0.01
    epochs = 5
    batchsize = 10
    ComparisonTraining1(MyNetwork,learningrate,epochs,batchsize)
    
""" 
    MyNetwork = PolicyNet()
    w=MyNetwork.weights
    learningrate = 0.01
    epochs = 5 #one epoch ~ 25 seconds
    TrainingBasicDan10(MyNetwork , learningrate, epochs)
    print("Faruk hats drauf")
    
    MyNetwork.weights = w
    learningrate = 0.05
    TrainingBasicDan10(MyNetwork,learningrate,epochs)
    print("Faruk geht ab")
    
"""

    
    
