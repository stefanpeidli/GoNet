# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:58:03 2017
@author: Stefan Peidli
"""
import numpy as np
import matplotlib.pyplot as plt
from Hashable import Hashable
from TrainingDataFromSgf import TrainingDataSgf
from TrainingDataFromSgf import TrainingDataSgfPass
from PolicyNet import PolicyNet
import time
import datetime

"""
This Script should be used to train Policy Networks.

Parameters: 
    I gave an advice for choosing reasonable values. Exceed those limits at your own risk. [minimum, recommendation, maximum]
    I gave an advice for choosing reasonable values. Exceed those limits at your own risk. 
    Format: [minimum, recommendation, maximum]
    
    learningrate (aka eta) : The "speed" of learning. Denotes how far we go into the direction of the gradient. [0, 0.01, 0.5]
    epochs : Number of epochs of learning. An epoch is finished, when all boards are forward and backward propagated once. [1, 3-10-30-100, 1000] On my PC with Dan_10 games (~4200 distinct board entries) 1 epoch took around 25 seconds, keep that in mind.
    sgfrange : when using all games in dgs, sgfrange denotes the number of game indices that should be added to the training data. [1, 300, 1 000 000] importing with range 10000 took 23 seconds. Propagating this might even take hours. Please use around 300 for a reasonable computation time and around multiple thousands of boards.
    batchsize : Data will be split into batchs of this size. Those will be propagatet as whole, then gradient descent is applied. If there is no option to choose batchsize, default is 1. [1, 100, size of board entries] requires further testing.
    trainingrate : Data will be split into a training set and test set by rate [trainingrate : (1-trainingrate)]. Training will be done on the trainingset, then the performance (error) on the testset is checked. [0, 0.8, 1] should heuristically be above 0.6
    tolerance : Error tolerance. Training will stop if error falls below it. [up to now, the error of 4 will not be reached that fast..., take what you like here between 0 and 3]
    maxepochs : Maximal amount of epochs to iterate. Training will stop once this epoch is reached. [1, 30, inf] keep the computation time in mind! in case, better run many small Trainings so the results are not lost!
    epochs : Number of epochs of learning. An epoch is finished, when all boards are forward and backward propagated once. 
    [1, 3-10-30-100, 1000] On my PC with Dan_10 games (~4200 distinct board entries) 1 epoch took around 25 seconds, 
    keep that in mind.
    sgfrange : when using all games in dgs, sgfrange denotes the number of game indices that should be added to the training data. 
    [1, 300, 1 000 000] importing with range 10000 took 23 seconds. Propagating this might even take hours. 
    Please use around 300 for a reasonable computation time and around multiple thousands of boards.
    batchsize : Data will be split into batchs of this size. Those will be propagatet as whole, then gradient descent is applied. 
    If there is no option to choose batchsize, default is 1. [1, 100, size of board entries] requires further testing.
    trainingrate : Data will be split into a training set and test set by rate [trainingrate : (1-trainingrate)]. 
    Training will be done on the trainingset, then the performance (error) on the testset is checked. 
    [0, 0.8, 1] should heuristically be above 0.6
    tolerance : Error tolerance. Training will stop if error falls below it. 
    [up to now, the error of 4 will not be reached that fast..., take what you like here between 0 and 3]
    maxepochs : Maximal amount of epochs to iterate. Training will stop once this epoch is reached. 
    [1, 30, inf] keep the computation time in mind! in case, better run many small Trainings so the results are not lost!

On saving and loading weights:
    FYI, weights can be loaded and saved into a folder < Saved_Weights > which is already in git.ignore. Please create this folder if it does not exist yet.
    FYI, weights can be loaded and saved into a folder < Saved_Weights > which is already in git.ignore. 
    Please create this folder if it does not exist yet.
    Let's say your trained your PolicyNet called Martha, then:
        Saving weights: Martha.saveweights(name)
        Loading weights: Martha.loadweightsfromfile(name)
    I also added a proposal for a weight naming format:
        name="weights"+datetime.datetime.now().strftime("%y%m%d%H%M")+"eta10000"+str(int(eta*10000))+"epochs"+str(epochs)+"batchsize"+"1"+"Dan10"
    I think this format is rather self-explaining. E.g.
        weights1712061818eta10000100epochs20batchsize1Dan10
        weights from 2017-12-06 at 18:18 with eta*10 000 = 100, epochs trained = 20, batchsize = 1, Dataset was Dan_10
        
        
"""

def TrainingBasic(PolicyNetwork, sgf_range = 1000, epochs=1, eta=0.01, batch_size=1, stoch_coeff=1, error_function=0):
    testdata = TrainingDataSgfPass("dgs",range(0,sgfrange))
    errors_by_epoch=PolicyNetwork.Learn(testdata, epochs, eta, batch_size, stoch_coeff, error_function)
    return errors_by_epoch

    
def TrainingSplit(PolicyNetwork, sgf_range = 1000, eta = 0.01, batch_size=1, stoch_coeff=1, error_function=0, trainingrate, error_tolerance, maxepochs):
    trainingdata = TrainingDataSgfPass("dgs",range(0,sgf_range))
    datasize = len(trainingdata.dic)
    [error,epochs] = PolicyNetwork.Learnsplit(trainingdata, eta, batch_size, stoch_coeff, error_function, trainingrate, error_tolerance, maxepochs)
    print("Datasize was",datasize,",Final K-L-Error:",error[-1:][0],",Epochs:",epochs)
    
    
def Training(PolicyNetwork, epochs=1, eta=0.01, batch_size=1, stoch_coeff=1, error_function=0, file = "dan_data_10")
    testset = TrainingDataSgfPass("dgs",file)
    t=time.time()
    init_error = PolicyNetwork.PropagateSet(testset,error_function)
    print("Propagation took",np.round(time.time()-t,3),"seconds.")
    print("Learning is done for",epochs,"epochs, with batch size",batch_size,",eta",eta,",stoch_coeff",stoch_coeff,",error_function number",error_function,"and with Games given by",file)
    t=time.time()
    errors_by_epoch=PolicyNetwork.Learn(testdata, epochs, eta, batch_size, stoch_coeff, error_function)
    finalerror = PolicyNetwork.PropagateSet(testset,error_function)
    print(finalerror,'Final Error:')
    print(initerror,'Initial Error:')
    print("total time needed:",time.time()-t)
    #save results:
    #name="weights"+datetime.datetime.now().strftime("%y%m%d%H%M")+"eta10000"+str(int(eta*10000))+"epochs"+str(epochs)+"batchsize"+"1"+"Dan10"
    #PolicyNetwork.saveweights(name)
    #print("weights have been saved to",name)
    return errors_by_epoch

"""
def ComparisonTraining1(PolicyNetwork,learningrate,epochs,batchsize):
    w=PolicyNetwork.weights
    testset = TrainingDataSgfPass("dgs","dan_data_10")
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
    PolicyNetwork.weights=w #reset weights, does not work...(?)
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
"""

# Training Area = The Neural Network Gym : Do training here
    
your_name = "Stefan"

# example for training:
if your_name is "Example":
    MyNetwork = PolicyNet()
    learningrate = 0.01
    epochs = 5 #one epoch ~ 25 seconds
    sgfrange = 10
    TrainingBasic(MyNetwork , learningrate, epochs, sgfrange)
    name="weights"+datetime.datetime.now().strftime("%y%m%d%H%M")+"eta10000"+str(int(learningrate*10000))+"epochs"+str(epochs)+"batchsize"+"1"+"sgfrange"+str(sgfrange)
    MyNetwork.saveweights(name)

# Paddy
if your_name is "Paddy":
    MyNetwork = PolicyNet()
    learningrate = 0.01
    epochs = 1  # one epoch ~ 25 seconds
    sgfrange = 10
    stoch_coeff = 0.2
    TrainingBasic(MyNetwork, learningrate, epochs, sgfrange, stoch_coeff)
    name = "weights" + datetime.datetime.now().strftime("%y%m%d%H%M") + "eta10000" + str(
        int(learningrate * 10000)) + "epochs" + str(epochs) + "batchsize" + "1" + "sgfrange" + str(sgfrange)
    MyNetwork.saveweights(name)


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
    
    if training_program == 2:
        PN=PolicyNet()
        PN.saveweights('test')
        epochs = 20
        print("I think I will need",np.round(epochs/3*(1+0.2+0.7),2),"minutes for this task.")
        errors_by_epoch1 = TrainingBasicDan10(PN,0.001,epochs,1)#vanilla grad descent
        PN.loadweightsfromfile('test')
        errors_by_epoch2 = TrainingBasicDan10(PN,0.001,epochs,0.2)
        PN.loadweightsfromfile('test')
        errors_by_epoch3 = TrainingBasicDan10(PN,0.001,epochs,0.7)
        plt.plot(range(0,epochs),errors_by_epoch1,'b')
        plt.plot(range(0,epochs),errors_by_epoch2,'g')
        plt.plot(range(0,epochs),errors_by_epoch3,'r')
        
"""


# Stefan:
if your_name is "Stefan":
    #hier schreibe ich mein training rein
    print("halo I bims")
    
    training_program = 1

    if training_program == 1:
        #3 epochs = 1 minute
        PN=PolicyNet()
        #PN.loadweightsfromfile('Saved_Weights','weights1712071159eta100001epochs60batchsize1Dan10')
        epochs=60
        print("I think I will need",np.round(epochs/3,2),"minutes for this task.")
        errors_by_epoch1 = TrainingHellingerDan10(PN,0.001,epochs,0.9)
        plt.plot(range(0,epochs),errors_by_epoch1)

    if training_program == 2:#TODO: check why saving the weights like this does not work
        PN=PolicyNet()
        w=PN.weights
        epochs = 30
        print("I think I will need",np.round(epochs/3*(1+0.2+0.7),2),"minutes for this task.")
        errors_by_epoch1 = TrainingBasicDan10(PN,0.001,epochs,1)
        PN.weights=w
        errors_by_epoch2 = TrainingBasicDan10(PN,0.001,epochs,0.2)
        PN.weights=w
        errors_by_epoch3 = TrainingBasicDan10(PN,0.001,epochs,0.7)
        plt.plot(range(0,epochs),errors_by_epoch1)
        plt.plot(range(0,epochs),errors_by_epoch2)
        plt.plot(range(0,epochs),errors_by_epoch3)
        
    if training_program == 3:
        PN=PolicyNet([9*9,100,200,300,400,300,200,100,9*9+1])
        epochs = 10
        
        errors_by_epoch1 = TrainingBasicDan10(PN,0.01,epochs,0.2)
        errors_by_epoch2 = TrainingBasicDan10(PN,0.001,epochs,0.5)
        errors_by_epoch3 = TrainingBasicDan10(PN,0.001,epochs,0.8)
        errors_by_epoch4 = TrainingBasicDan10(PN,0.0005,epochs,0.95)
        
        name="weightsdelight"
        PN.saveweights(name)
        plt.plot(range(0,epochs*4),[errors_by_epoch1,errors_by_epoch2,errors_by_epoch3,errors_by_epoch4])
