# -*- coding: utf-8 -*-
"""
Created on Tue May  9 23:23:07 2017

@author: Kyriakis Dimitrios
"""


import random as rd
import numpy as np
import math as mt
from scipy import stats
from scipy.spatial import distance

######################## FUNCTIONS  ###########################################



######################## Metric Distance  #####################################
def Dist_func(x,y,Distance,S):
    D = x.shape[0]
    if Distance =="1":
        Eucl_func =  lambda x,y: np.linalg.norm(x.reshape(D,1)-y.reshape(D,1))
        dist = Eucl_func(x,y)
    elif Distance =="2":
        Manha_func = lambda x,y: abs(np.sum(x.reshape(D,1)-y.reshape(D,1)))
        dist = Manha_func(x,y)
    elif Distance =="3":    
        Mahala_func =  lambda x,y: np.sqrt((x-y).reshape(1,D).dot(np.linalg.inv(S)).dot((x.reshape(D,1)-y.reshape(D,1)).reshape(D,1)))[0][0]

        dist = Mahala_func(x,y)        
    elif Distance =="4":
        dist = distance.cdist(x.reshape(1,D),y.reshape(1,D),"cosine")
    elif Distance == "5":
        dist = distance.cdist(x.reshape(1,D),y.reshape(1,D),"correlation")
    elif Distance == "6":
        dist = distance.cdist(x.reshape(1,D),y.reshape(1,D),'chebyshev')
    return dist


#-----------------------------------------------------------------------------#
###============================ CREATE TEST SET ============================###
#-----------------------------------------------------------------------------#

def Create_Test(X,labels,full_indexes):
    '''
    **Description:** Hide the 10% of the data\n
    **Input:**\n
    - X: all data\n
    - Labels: all labels (and the hidden)\n
    **Output:**\n
    - Test indexes\n
    - Train indexes (all indexes without test)\n
    '''
    n_samples = len(labels)
    labels_new=[]
    while len(set(labels)) !=  len(set(labels_new)):
        Test_indexes = rd.sample(full_indexes,round(0.1*n_samples))
        labels_new = labels[list(set(full_indexes) - set(Test_indexes))]
    
    Train_indexes = list(set(full_indexes) - set(Test_indexes))
    return Test_indexes,Train_indexes


#------------------------------------------------------------#
###================== ACCURACY ======================###
#------------------------------------------------------------#
def Accuracy(Predicted,Test_indexes,Original):
    counter =0
    for index in range(len(Predicted)):
        if Predicted[index] == Original[Test_indexes[index]]:
            counter +=1
    ### PRINT PRFEDICTED AND ORIGINALK LABELS
#    for j in range(len(Predicted)):
#        print(Predicted[j],Original[Test_indexes[j]])
    return round(counter/len(Predicted),2)*100


#------------------------------------------------------------#
###================== STRATIFICATION ======================###
#------------------------------------------------------------#
def Stratification(X,Train_Labels):
    '''
    **Description:** Calculate the probability of each class, indexies, mean , std\n
    - *Labels*: List with label names
    '''
    Percent_dic ={}
    mini = mt.inf
    for label in set(Train_Labels):
        count_label = list(Train_Labels).count(label)
        # calcilate the rare label
        mini = min(mini,count_label)
        Class_indexies = list(np.where(Train_Labels == label)[0])
        mean = np.mean(X[:,Class_indexies],axis = 1)
        std = np.std(X[:,Class_indexies],axis = 1)
        # list indexies per label
        Percent_dic[label] = [count_label,round(count_label/len(Train_Labels),3),Class_indexies,mean,std]
    return Percent_dic, mini



#------------------------------------------------------------#
###=============== CREATE FOLDS ===========================###
#------------------------------------------------------------#
def Create_Folds (Percent_dic,Train_indexes,mini):
    '''
    **Description:** Split the data in Folds\n
    **Input:** \n
    - Stratified Matrix\n
    - Size of folds\n
    - Mini = # of samples of the most rare Class\n
    **Output:** \n
    - *Folds*: List with list of indexes for every fold
    '''
    def_dict_per = Percent_dic.copy()
    size_of_folds = round(len(Train_indexes)/mini)
    folds = []
    for i in range(mini):
        indexes_per_fold =[]
        for j in Percent_dic.keys():
            if i < mini-1:
                per_list = rd.sample(def_dict_per[j][2],round(def_dict_per[j][1]*size_of_folds))
            # Take the rest in last one
            else :
                per_list = def_dict_per[j][2]
            indexes_per_fold+=per_list
            # Delete so we cannot select
            def_dict_per[j][2] = list(set(def_dict_per[j][2]) - set(per_list))
        folds.append(indexes_per_fold)
    return folds

def Cross_Validation( X, Folds, Train_indexes, All_Labels, number_of_neighbours, Method, Dist_func, Distance,S):
    accuracy = 0
    for fold in Folds:
        valid_indx = fold
        valid_data = X[:,valid_indx].T       
        train_indx = list(set(Train_indexes) - set(fold))
        train_data = X[:,train_indx].T
        Percent_dic,mini_fold = Stratification(X,All_Labels[train_indx])
        # ORDER LABELS AND PROBABILITIES
        Labels_list = []
        Percentage_ci_list = []
        for label in Percent_dic.keys():
            Labels_list.append(label)
            Percentage_ci_list.append( Percent_dic[label][1])

        
        #====================================================#
        #################  CHOOSE METHOD #####################
        #====================================================#
        if Method == "2":
            predict_labels = Naive(X,valid_indx,All_Labels,Labels_list,Percentage_ci_list,Percent_dic)
            
        elif Method == "1":
            ## DISTANCE MATRIX ##
            Dist_Matrix = np.zeros((valid_data.shape[0],train_data.shape[0]))
            
            for  i  in range(valid_data.shape[0]):
               for j in range(train_data.shape[0]):
                   Dist_Matrix[i,j] = Dist_func(train_data[j,:],valid_data[i,:],Distance,S)
            Matrix = Dist_Matrix.copy()
            ## ACCURACY TEST ##
            predict_labels = K_nn(Matrix,All_Labels[train_indx],All_Labels[valid_indx],number_of_neighbours)
    
        fold_accuracy = Accuracy(predict_labels,valid_indx,All_Labels)
        accuracy += fold_accuracy
    
    return round(accuracy/len(Folds),2)

#------------------------------------------------------------#
###================== POSTERIOR ===========================###
#------------------------------------------------------------#

def Posterior_Prob (X,Vector,Labels_list, All_Labels,Percent_dic, Percentage_ci_list):
    '''
    Description: Find the Posterior Probability\n
    - Percent_dicnt_dic: Dictionary with keys the calsses and values the # counts, percentage, indexies, mean, std\n
    - Vector: A sample with features
    '''
    Prob_Matrix = np.zeros((len(Vector),len(Percent_dic.keys())))
    
    for clash in range(len(Labels_list)):
        for i in range(len(Vector)):
            mean = Percent_dic[Labels_list[clash]][3][i]
            sdv = Percent_dic[Labels_list[clash]][4][i]
            
#            if sdv == 0:
#                if Vector[i] == mean:
#                    cond_prob = 4
#                else:
#                    cond_prob = 0
#            else:
            cond_prob = stats.norm.pdf(Vector[i],loc=mean,scale=sdv)
#               cond_prob = (1/(np.sqrt(2*np.pi)))*(1/sdv)*np.exp((-1/2)*((Vector[i]-mean)**2)/(sdv**2))
       
            ####### ZERO conditional probability ###
            if  mt.isnan(cond_prob) or cond_prob == 0 :   # cond_prob < 1e-20 or
                # list of train for a feature
                xj_ci = list(X[i,Percent_dic[Labels_list[clash]][2]])
#                xj_ci = list(X[i,Percent_dic[Labels_list[clash]][2]])
                nc =0 
                for js in xj_ci:
                    if js >= Vector[i]-sdv and js<= Vector[i]+sdv:
                        nc+=1
#                nc = xj_ci.count(Vector[i])
                n = Percent_dic[Labels_list[clash]][0]
                m =1
                p = 1/len(set(xj_ci))
                cond_prob = (nc + (m*p)) / (n + m)
                
            ################################################
            
            Prob_Matrix[i,clash] = cond_prob

    product_array = np.prod(Prob_Matrix, axis = 0)
    max_posterior_indx = np.argmax(product_array *Percentage_ci_list)
    predicted_label = Labels_list[max_posterior_indx]
    return predicted_label


def Naive(X, Test_indexes, All_Labels, Labels_list, Percentage_ci_list,Percent_dic):
    predicted_test =[]
    Test_data = X[:,Test_indexes]
    for num_sample in range(len(Test_indexes)):
        Sample_Vector = Test_data[:,num_sample]
        predict = Posterior_Prob(X, Sample_Vector, Labels_list, All_Labels, Percent_dic, Percentage_ci_list)
        predicted_test.append(predict)
    return predicted_test

def K_nn(Dist_Matrix,train_labels,valid_labels,num_of_neigh):
    predicted =[]
    for i in range(Dist_Matrix.shape[0]):
        k_list = []
        for k in range(num_of_neigh):
            min_dist_indx =list(Dist_Matrix[i,]).index(min(list(Dist_Matrix[i,])))
            Dist_Matrix[i,min_dist_indx] = mt.inf
            k_list.append(train_labels[min_dist_indx])
        vote_dic={}
        for k in k_list:
            if k not in vote_dic.keys():
                vote_dic[k] = 1
            else:
                vote_dic[k]+=1
        ### IF THERE IS A TIE TAKE THE CLOSEST NEIGHBOUR
        predict_label = [key for key,val in vote_dic.items() if val == max(vote_dic.values())][0]
        predicted.append(predict_label) 
    return  predicted
