# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:58:03 2017
@author: Stefan Peidli
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

from Hashable import Hashable
import Board
from TrainingDataFromSgf import TrainingDataSgfPass
from PolicyNet import PolicyNet
import time
import datetime
import sqlite3

"""
This Script should be used to train Policy Networks.

Parameters: 
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


"""    
def training_split(PolicyNetwork, trainingrate, error_tolerance, maxepochs, sgf_range = 1000, eta = 0.01, batch_size=1, stoch_coeff=1, error_function=1, activation_function=0):
    trainingdata = TrainingDataSgfPass("dgs",range(0,sgf_range))
    datasize = len(trainingdata.dic)
    [error,epochs] = PolicyNetwork.Learnsplit(trainingdata, eta, batch_size, stoch_coeff, error_function, activation_function, trainingrate, error_tolerance, maxepochs)
    print("Datasize was",datasize,",Final K-L-Error:",error[-1:][0],",Epochs:",epochs)
"""

    
def training(PolicyNetwork, epochs=10, eta=0.001, batch_size=5, error_function=0, file="dan_data_10",
             adaptive_rule='linear', sample_proportion = 1, db=False, db_name="none", details=True):
    t = time.time()
    if not db:
        testset = TrainingDataSgfPass("dgs", file)
        init_error = PolicyNetwork.propagate_set(testset, db, adaptive_rule, error_function)
    else:
        [no, sett] = PolicyNetwork.extract_batches_from_db(db_name, batch_size, sample_proportion)
        testset = [no, sett]
        init_error = "TODO"  # TODO do this. Problem: Sett might not contain all games (sample-prop!=1)

    if details:
        print("Propagation and import of the set took", np.round(time.time()-t, 3), "seconds.")
        print("Learning is done for", epochs, "epochs, with batch size", batch_size, ",eta", eta,
              ",error function number", error_function, "and with games given by the file ", file, ".")
        t = time.time()
        print("Learning in progress...")
    
    errors_by_epoch = PolicyNetwork.learn(testset, epochs, eta, batch_size, sample_proportion,
                                          error_function, db, db_name, adaptive_rule)

    if not db:
        testset = TrainingDataSgfPass("dgs", file)
        final_error = PolicyNetwork.propagate_set(testset, db, adaptive_rule, error_function)
    else:
        final_error = "TODO"  # TODO do this
    if details:
        print("Finished learning.")
        print("Details on the results:")
        print('Initial Error:', init_error)
        print('Final Error:', final_error)
        print("Total time needed for training:", time.time()-t)

        print("Visualization:")
        plt.figure(0)
        plt.plot(range(0, len(errors_by_epoch)), errors_by_epoch)
        plt.title("Error in each epoch")
        plt.xlabel("epochs")
        plt.ylabel("Error")
        print("Error was measured with error function number", str(error_function))
        plt.show()

    return errors_by_epoch


def train_dict(layers=[9 * 9, 1000, 200, 9 * 9 + 1], filter_ids=[0, 1, 2, 3, 4, 5, 6, 7], batch_size=100, eta=0.001,
               err_fct=0, duration_in_hours=8, custom_save_name="none", adaptive_rule="logarithmic"):
    print("This Script will generate a PolicyNet and train it for a certain time.")
    print("For this, Dictionaries are used. No DBs are involved.")
    print("")
    print("Info:")
    print("Layers ", layers)
    print("filter_ids ", filter_ids)
    print("batch_size", batch_size)
    print("eta", eta)
    print("err_fct", err_fct)
    print("duration_in_hours", duration_in_hours)
    print("custom_save_name", custom_save_name)
    print("adaptive_rule", adaptive_rule)
    print("")
    PN = PolicyNet(layers=layers, filter_ids=filter_ids)
    testset = TrainingDataSgfPass("dgs", "dan_data_10")
    print("Games have been imported from dan_data_10.")
    [number_of_batchs, batchs] = PN.splitintobatches(testset, batch_size)
    print("Split up into", number_of_batchs, "Batches with size", batch_size, ".")
    errors_by_epoch = []
    init_error = PN.propagate_set(testset, False, adaptive_rule, err_fct)
    start = time.time()
    epoch = 0
    print("Training process starts now. It will take ", duration_in_hours, " hours.")
    while time.time() - start < duration_in_hours * 60 * 60:
        t = time.time()
        errors_by_epoch.append(0)
        for i_batch in range(0, number_of_batchs):
            error_in_batch = PN.learn_batch(batchs[i_batch], eta, err_fct, False, adaptive_rule, True)
            errors_by_epoch[epoch] += error_in_batch
        errors_by_epoch[epoch] = errors_by_epoch[epoch] / number_of_batchs
        print("Epoch", epoch, "with error", errors_by_epoch[epoch])
        print("Time needed for epoch in seconds:", np.round(time.time() - t))
        epoch = epoch + 1
    print("")
    if custom_save_name is "none":
        save_name = "weights"+str(duration_in_hours)+"hours"+"".join(str(x) for x in filter_ids)+"filtids"+epoch+"epochs"
    else:
        save_name = custom_save_name
    PN.saveweights(save_name)
    total_time = time.time() - start
    print("Total time taken for training:", total_time, "and epochs", epoch)
    print("Average time per epoch:", total_time / epoch)
    print("Initial error:", init_error)
    print("Final error:", errors_by_epoch[-1])
    improvement = init_error - errors_by_epoch[-1]
    print("Total error improvement:", improvement)
    print("Error development: ", errors_by_epoch)
    print("Error reduction per second:", improvement / total_time)
    plt.plot(range(0, len(errors_by_epoch)), errors_by_epoch)
    plt.show()


def train_db(layers=[9 * 9, 1000, 200, 9 * 9 + 1], filter_ids=[0, 1, 2, 3, 4, 5, 6, 7], batch_size=200, eta=0.001,
             err_fct=0, epochs = 0, duration_in_hours = 0.1, sample_proportion = 1, db_name="dan_data_10",
             custom_save_name="none", adaptive_rule="none", db_move = True):
    print("This Script will generate a PolicyNet and train it for a certain time.")
    print("For this, DBs are used. Static: This means we will keep the batches once created, else we would rebuild it.")
    print("If and only if you don't choose an adaptive rule, Board duplicates will be used.")
    print("This Script ONLY works with Absolute board distributions, not dirac!. So we use the 'dist' DB folder.")
    print()
    print("Info:")
    print("Layers ", layers)
    print("filter_ids ", filter_ids)
    print("batch_size", batch_size)
    print("eta", eta)
    print("err_fct", err_fct)
    print("duration_in_hours", duration_in_hours)
    print("custom_save_name", custom_save_name)
    print("adaptive_rule", adaptive_rule)
    print("")

    PN = PolicyNet(layers=layers, filter_ids=filter_ids)
    print("Games have been imported from " + db_name)
    if not db_move:
        con = sqlite3.connect(r"DB/Dist/" + db_name, detect_types=sqlite3.PARSE_DECLTYPES)
    else:
        con = sqlite3.connect(r"DB/Move/" + db_name, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("select count(*) from movedata")
    datasize = int(np.ceil(cur.fetchone()[0] * sample_proportion))
    con.close()
    if custom_save_name is "none":
        save_name = "weights" + str(duration_in_hours) + "hours" + "".join(
            str(x) for x in filter_ids) + "filtids" + str(epochs) + "epochs"
    else:
        save_name = custom_save_name

    errors_by_epoch = []
    whole_id_set = PN.gen_id_list_from_db(db_name, datasize, sample_proportion, db_move)[1]
    whole_set = PN.gen_whole_set_from_id_list(whole_id_set, db_name, db_move)
    init_error = PN.propagate_set(whole_set, True, adaptive_rule, err_fct)  # Error needs to be measured on whole set
    [number_of_batches, batch_id_list] = PN.gen_id_list_from_db(db_name, batch_size, sample_proportion, db_move)
    print("Split up into", len(batch_id_list), "Batches with size", batch_size, ".")
    start = time.time()
    if (epochs == 0 and duration_in_hours == 0) or (epochs != 0 and duration_in_hours != 0):
        print("")
        print("please choose a feasible combination of epochs and duration_in_hours: depending on which of the two"
              "parameters is >0 you'll start a training for a certain time period or a count of epochs")
        return
    elif epochs == 0:
        epoch = 0
        print("Training process starts now. It will take ", duration_in_hours, " hours.")
        while time.time() - start < duration_in_hours * 60 * 60:
            epoch += 1
            t = time.time()
            errors_by_epoch.append(0)
            batches = PN.extract_batches_from_id_list(number_of_batches, batch_id_list, db_name, db_move)[1]
            for i_batch in range(number_of_batches):
                error_in_batch = PN.learn_batch(batches[i_batch], eta, err_fct, True, adaptive_rule, True)
                errors_by_epoch[epoch-1] += error_in_batch
            errors_by_epoch[epoch-1] = errors_by_epoch[epoch-1] / number_of_batches
            print("Epoch", epoch, "with error", errors_by_epoch[epoch-1])
            print("Time needed for epoch in seconds:", np.round(time.time() - t))
            if epoch%20 == 0:
                PN.saveweights(save_name)
    else:
        epoch = 1
        print("Training process of " + str(epochs) + " epochs will start now")
        for epoch in range(1,epochs+1):
            errors_by_epoch.append(0)
            batches = PN.extract_batches_from_id_list(number_of_batches, batch_id_list, db_name, db_move)[1]
            for i_batch in range(number_of_batches):
                error_in_batch = PN.learn_batch(batches[i_batch], eta, err_fct, True, adaptive_rule, True)
                errors_by_epoch[epoch-1] += error_in_batch
            errors_by_epoch[epoch-1] = errors_by_epoch[epoch-1] / number_of_batches
            print("Epoch", epoch, "with error", errors_by_epoch[epoch-1])
            if epoch%20 == 0:
                PN.saveweights(save_name)
    print("")
    PN.saveweights(save_name)
    total_time = time.time() - start
    final_error = PN.propagate_set(whole_set, True, adaptive_rule, err_fct)  # Error needs to be measured on whole set
    print("Total time taken for training:", total_time, "and epochs", epoch)
    print("Average time per epoch:", total_time / epoch)
    print("Initial error:", init_error)
    print("Final error:", final_error)
    improvement = init_error - final_error
    print("Total error improvement:", improvement)
    print("Error development: ", errors_by_epoch)
    print("Error reduction per second:", improvement / total_time)
    plt.plot(range(0, len(errors_by_epoch)), errors_by_epoch)
    plt.show()

    return [total_time, init_error, final_error, epoch]

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
    
your_name = "Beno"

# example for training:
if your_name is "Example":
    MyNetwork = PolicyNet()
    eta = 0.01
    epochs = 5
    sgf_range = 10
    batch_size = 5
    error_function = 0  # KLD
    training_basic(MyNetwork, sgf_range, epochs, eta, batch_size, error_function)
    name = "weights"+datetime.datetime.now().strftime("%y%m%d%H%M")+"eta10000"+str(int(eta*10000))+"epochs"+\
           str(epochs)+"batchsize"+str(batch_size)+"sgfrange"+str(sgf_range)
    MyNetwork.saveweights(name)

# Paddy
if your_name is "Paddy":
    MyNetwork = PolicyNet([9*9,1000,9*9+1])
    epochs=10
    sample_proportion=1
    error_function=0
    TrainingAdvanced(MyNetwork, epochs, sample_proportion, error_function)
    learningrate=0.01
    name = "weights" + datetime.datetime.now().strftime("%y%m%d%H%M") + "eta10000" + str(
        int(learningrate * 10000)) + "epochs" + str(epochs) + "batchsize" + "1"
    MyNetwork.saveweights(name)

# Beno
if your_name is "Beno":
    layers = [9*9, 784, 9*9 + 1]
    filter_ids = [6, 7, 8]
    batch_size = 10
    eta = 0.001
    error_function = 0
    [epochs,duration_in_hours] = [240,0]
    sample_proportion = 0.005
    db_move = True
    db_name = 'data_3'
    custom_save_name = 'small_batches_test_2'
    adaptive_rule = 'none'
    [total_time, init_error, final_error, last_epoch] = train_db(layers, filter_ids, batch_size, eta, error_function, epochs,
                                                     duration_in_hours, sample_proportion, db_name, custom_save_name=
                                                     custom_save_name, adaptive_rule=adaptive_rule, db_move=db_move)
    if epochs == 0:
        seconds_per_epoch =  total_time / last_epoch
    else:
        seconds_per_epoch = total_time/epochs
    fclient = open('logs/'+custom_save_name, 'w')
    fclient.write('Training Session ' + custom_save_name + '\n' + '...\n' + 'filter_ids ' + str(filter_ids) + '\n'
                  + 'batch_size ' + str(batch_size) + '\n' + 'eta ' + str(eta) + '\n' + 'errof_function '
                  + str(error_function) + '\n' + '[epochs, duration]: ' + str([epochs, duration_in_hours]) + '\n'
                  + 'sample_proportion ' + str(sample_proportion) + '\n' + 'moveDB ' + str(db_move) + '\n' + 'db_name: '
                  + db_name + '\n' + 'adaptive rule: ' + adaptive_rule + '\n' + 'total time: ' + str(total_time) + '\n'
                  + 'init error, final error: ' + str([init_error, final_error]) + '\n' + 'error reduction overall: '
                  + '\n' + str(final_error-init_error)  + '\n' + 'error reduction per second: ' + '\n'
                  + str((final_error - init_error)/total_time) + '\n' + 'seconds per epoch: ' + str(seconds_per_epoch))

# Stefan:
if your_name is "Stefan":
    # hier schreibe ich mein training rein
    print("halo I bims")
    
    training_program = 9
    
    if training_program == 1:  # Checking error on a testset, while training on a different set
        PN=PolicyNet()
        testset = TrainingDataSgfPass("dgs","dan_data_10")
        trainingset = TrainingDataSgfPass("dgs","dan_data_295")
        start=time.time()
        errors_by_training=[]
        while time.time() - start < 0.5 * 60 * 60: #half hour
            errors_by_epoch=PN.Learn(trainingset,5,0.01,200,0.8,0)
            errors_by_training.append(errors_by_epoch)
        
    if training_program == 2: #standard batch Dan 10
        trainingdata = TrainingDataSgfPass("dgs","dan_data_10")
        batch_size = 100
        eta = 0.01
        stoch_coeff = 1
        epochs = 20
        error_function = 0
        activation_function = 0
        PN = PolicyNet([9*9,500,9*9+1],activation_function)
        
        errors=[]
        print(PN.PropagateSet(trainingdata,error_function))
        start=time.time()
        
        [number_of_batchs, batchs] = PN.splitintobatches(trainingdata,batch_size)
        errors_by_epoch = []
        for epoch in range(0,epochs):
            errors_by_epoch.append(0)
            for i_batch in range(0,number_of_batchs):
                error_in_batch = PN.LearnSingleBatch(batchs[i_batch], eta, stoch_coeff, error_function)
                errors_by_epoch[epoch] += error_in_batch
            errors_by_epoch[epoch] = errors_by_epoch[epoch] / number_of_batchs
            print (errors_by_epoch[epoch])
        print(time.time()-start)
        plt.plot(range(0,len(errors_by_epoch)),errors_by_epoch)
        #name="GOTEST_Number_"+str(num)
        #num+=1
        #PN.saveweights(name)
            
    if training_program == 3: #standard batch Dan 295
        trainingdata = TrainingDataSgfPass("dgs","dan_data_295")
        batch_size = 1000
        eta = 0.01
        stoch_coeff = 1
        error_function = 0 #0 => KLD
        activation_function = 0 #0 => tanh sigmoid
        PN = PolicyNet([9*9,1000,9*9+1],activation_function)
        
        errors=[]
        print(PN.PropagateSet(trainingdata,error_function))
        start=time.time()
        
        [number_of_batchs, batchs] = PN.splitintobatches(trainingdata,batch_size)
        errors_by_epoch = []
        while time.time()-start<8*60*60: #8 stunden
            errors_by_epoch.append(0)
            for i_batch in range(0,number_of_batchs):
                error_in_batch = PN.LearnSingleBatch(batchs[i_batch], eta, stoch_coeff, error_function)
                errors_by_epoch[-1] += error_in_batch
            errors_by_epoch[-1] = errors_by_epoch[-1] / number_of_batchs
            if len(errors_by_epoch)>10:
                improvement_in_10_epochs = np.abs(errors_by_epoch[-1]-np.sum(errors_by_epoch[-10:])/10)
                if improvement_in_10_epochs <0.00001:
                    eta=eta*0.9 #refine step size if we do not improve
                    print("Eta decreased to",eta)
            print (len(errors_by_epoch),errors_by_epoch[-1])
        
        plt.plot(range(0,len(errors_by_epoch)),errors_by_epoch)
        #PN.saveweights("AmbitiousLearning")
        print("finished with KLD of",errors_by_epoch[-1],", a total of",len(errors_by_epoch),"epochs and a final eta of ",eta)
        print("Time needed:",time.time()-start)
        
    if training_program == 4: #ambitious
        trainingdata = TrainingDataSgfPass("dgs","dan_data_295")
        batch_size = 1000
        eta = 0.01
        stoch_coeff = 1
        error_function = 0 #0 => KLD
        activation_function = 0 #0 => tanh sigmoid
        PN = PolicyNet([9*9,1000,9*9+1],activation_function)
        PN.loadweightsfromfile("AmbitiousLearning")
        
        print(PN.PropagateSet(trainingdata,error_function))
        
    if training_program == 5:#adaptive eta
        trainingdata = TrainingDataSgfPass(folder="dgs",id_list="dan_data_10")
        batch_size = 100
        eta = 0.01
        stoch_coeff = 1
        epochs = 20
        error_function = 0
        activation_function = 0
        PN = PolicyNet([9*9,500,9*9+1],activation_function)
        
        errors=[]
        print(PN.PropagateSet(trainingdata,error_function))
        start=time.time()
        
        [number_of_batchs, batchs] = PN.splitintobatches(trainingdata,batch_size)
        errors_by_epoch = []
        adaptive_errors=[]
        for epoch in range(0,epochs):
            errors_by_epoch.append(0)
            for i_batch in range(0,number_of_batchs):
                error_in_batch = PN.LearnSingleBatchAdaptive(batchs[i_batch], eta, stoch_coeff, error_function)
                errors_by_epoch[epoch] += error_in_batch
            errors_by_epoch[epoch] = errors_by_epoch[epoch] / number_of_batchs
            adaptive_errors.append(PN.PropagateSetAdaptive(trainingdata,error_function))
            print ("Error ",np.round(errors_by_epoch[epoch],5),"in epoch",epoch)
            print ("Adaptive error ",np.round(adaptive_errors[epoch],5),"in epoch",epoch)
        print(time.time()-start)
        plt.figure(0)
        plt.plot(range(0,len(errors_by_epoch)),errors_by_epoch)
        plt.plot(range(0,len(adaptive_errors)),adaptive_errors)
        
        #test for empty board
        for entry in trainingdata.dic:
            if all(PN.convert_input(Hashable.unwrap(entry))==0.45):
                zerotarg1 = trainingdata.dic[entry].reshape(9*9+1)
                zerotarg=zerotarg1/np.sum(zerotarg1)
        b=Board.Board(9)
        y=PN.Propagate(b)
        plt.figure(1)
        plt.plot(range(len(zerotarg)),zerotarg)
        plt.plot(range(len(y)),y)

    if training_program == 6:
        PN = PolicyNet(filter_ids=[0, 1])
        epochs = 10
        eta = 0.001
        batch_size = 100
        error_function = 0
        file = "dan_data_10"
        adaptive_rule = 'none'
        db = False
        db_name = "none"
        enrichment = False
        details = True
        training(PN, epochs, eta, batch_size, error_function, file, adaptive_rule, db=False, details=details)

    if training_program == 7:  # with dbs
        PN = PolicyNet(filter_ids=[0, 1, 2, 3, 4, 5, 6, 7])
        epochs = 5
        eta = 0.001
        batch_size = 100
        error_function = 0
        file = "dan_data_10"
        adaptive_rule = 'linear'
        sample_proportion = 1
        details = True
        training(PN, epochs, eta, batch_size, error_function, file, adaptive_rule, sample_proportion, True,
                 "dan_data_10", details=details)

    if training_program == 8:  # dict with adaptive eta rule.
        train_dict(layers=[9 * 9, 1000, 200, 9 * 9 + 1], filter_ids=[0, 1, 2, 3, 4, 5, 6, 7], batch_size=100, eta=0.001,
                   err_fct=0, duration_in_hours=20/60, custom_save_name="TEST8", adaptive_rule="logarithmic")

    if training_program == 9:  # db with duplicates
        train_db(layers=[9 * 9, 1000, 200, 9 * 9 + 1], filter_ids=[0, 1, 2, 3, 4, 5, 6, 7], batch_size=200,
                 eta=0.005, err_fct=0, duration_in_hours=10/60, custom_save_name="TEST9", adaptive_rule="none",
                 static=True)

    if training_program == 10:  # db without duplicates, i.e. with adaptive eta rule.
        train_db_static(layers=[9 * 9, 1000, 200, 9 * 9 + 1], filter_ids=[0, 1, 2, 3, 4, 5, 6, 7], batch_size=100,
                        eta=0.001, err_fct=0, duration_in_hours=20/60, custom_save_name="TEST10", adaptive_rule="logarithmic")
