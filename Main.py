# -*- coding: utf-8 -*-
"""
Created on Tue May  9 18:20:01 2017

@author: Kyriakis Dimitrios
"""
from Functions import *
import numpy as np
############## PREPARE OUR DATA SET ##########################

## STAGE 1
X = np.loadtxt("yeast.data.txt",usecols= (1,2,3,4,5,6,7,8)).T
print(X)
samples = np.loadtxt("yeast.data.txt",usecols = [0],dtype=str)
All_Labels = np.loadtxt("yeast.data.txt",usecols = [9],dtype=str)
D = X.shape[0]
N = X.shape[1]
N_samples = X.shape[1]

## INDEXIES ###
Full_indexes = list(range(N_samples))
Labels_set = list(set(All_Labels))

########### TEST SET #############
##### Check if class is deleted ##

Test_indexes, Train_indexes = Create_Test(X,All_Labels, Full_indexes)
print(X[:,Train_indexes])


############################## NORMALIZATION ##################################
MEAN = np.mean(X[:,Train_indexes], axis =1)
MAX = np.max(X[:,Train_indexes], axis =1)
MIN = np.min(X[:,Train_indexes], axis =1)
SUB = MAX - MIN

print(MEAN.shape)
STD = np.std(X[:,Train_indexes], axis =1)
Mean_Matrix = np.repeat(MEAN.reshape(D,1),len(Train_indexes),axis =1)
Min_Matrix = np.repeat(MIN.reshape(D,1),len(Train_indexes),axis =1)
STD_Matrix = np.repeat(STD.reshape(D,1),len(Train_indexes),axis =1)
SUB_Matrix = np.repeat(SUB.reshape(D,1),len(Train_indexes),axis =1)
X[:,Train_indexes] = (X[:,Train_indexes] - Min_Matrix)/SUB_Matrix


X[:,Test_indexes] = (X[:, Test_indexes] - np.repeat(MIN.reshape(D,1),len(Test_indexes),axis =1))/np.repeat(SUB.reshape(D,1), len(Test_indexes),axis =1)


S = np.cov(X[:,Train_indexes])
#######################################################################
##################### PLOT DATA ######################
#from sklearn.decomposition import PCA as pca
#Train_Matrix = X[:,Train_indexes]
#Test_data = X[:,Test_indexes]
#print(Train_Matrix.shape)
#n_components = 3
#my_pca = pca(n_components)
#Projected_Data = my_pca.fit_transform(Train_Matrix.T)
#print(Projected_Data.shape)
#print(len(All_Labels[Train_indexes]))
#import matplotlib.pyplot as plt
#from sklearn.metrics import silhouette_score
#from matplotlib import cm as cm
#
#x_1 = list(Projected_Data[:,0])
#y_1 = list(Projected_Data[:,1])
#
#fig2, bx = plt.subplots()
#colors = cm.rainbow(np.linspace(0,1,len(Labels_set)))
#
#for i,j in zip(range(Projected_Data.shape[0]),All_Labels[Train_indexes]):
#    bx.scatter(x_1[i], y_1[i],color = colors[Labels_set.index(j)])
#
#x_1 = list(Projected_Data[:,0])
#y_1 = list(Projected_Data[:,2])
#
#fig2, cx = plt.subplots()
#for i,j in zip(range(Projected_Data.shape[0]),All_Labels[Train_indexes]):
#    cx.scatter(x_1[i], y_1[i],color = colors[Labels_set.index(j)])
#
#
#
#x_1 = list(Projected_Data[:,1])
#y_1 = list(Projected_Data[:,2])
#
#fig2, dx = plt.subplots()
#for i,j in zip(range(Projected_Data.shape[0]),All_Labels[Train_indexes]):
#    dx.scatter(x_1[i], y_1[i],color = colors[Labels_set.index(j)])
#
#plt.show()



#=============================================================================#
############################  STRATIFICATION ##################################
#=============================================================================#
Percent_dic,mini = Stratification(X,All_Labels[Train_indexes])



#=============================================================================#
#################### ORDER LABELS AND PROBABILITIES ###########################
#=============================================================================#

Labels_list = []
Percentage_ci_list = []
for class_label in Percent_dic.keys():
    Labels_list.append(class_label)
    Percentage_ci_list.append(Percent_dic[class_label][1])

#=============================================================================#
############################### CREATE FOLDS ##################################
#=============================================================================#
Methods  = input("###############\n Choose Method \n###############  \n\t 1. Cross Validation\n\t 2. Leave one Out\n(1/2): ")
if Methods == "1":
    num_fold = int(input("Choose number of folds (Maximum 5 is suggested) : "))
    if num_fold > mini:
        num_fold = mini
    Folds = Create_Folds(Percent_dic,Train_indexes,num_fold)
elif Methods == "2":
    Folds = [[x] for x in range(X.shape[1])]

#=============================================================================#
################################# Parameters ##################################
#=============================================================================#



Method =input("Choose Classification Method:\n\t1. K-NN \n\t2. Naive\n(1/2): ")
if Method == "1":
    Neighbour_list = input("###########################\nChoose Number of Neighbours\n###########################\n(2,3,4,..) : ").split(",")
    Distance =input("""Choose metric distance:\n
    \t1. Euclidean\n
    \t2. Manhatan\n
    \t3. Mahalanobis\n
    \t4. Cosine\n
    \t5. Correlation\n
    \t6. Chebyshev\n
    (1/2/3/4/5/6): """)
 
#===========================1=================================================#
###==================== CALCULATE ACCURACY IN FOLDS  =======================###
#=============================================================================#
if Method == "1":
    for number_of_neighbours in Neighbour_list:
        num_of_neig = int(number_of_neighbours)
        if Methods =="1":
            Accuracy_Estimation = Cross_Validation( X, Folds, Train_indexes, All_Labels, num_of_neig, Method, Dist_func, Distance,S)
        elif Methods == "2":  
            ########   LEAVE ONE OUT ###############
            
            Accuracy_Estimation = Cross_Validation( X, Folds, Train_indexes, All_Labels, num_of_neig, Method, Dist_func, Distance,S)
        print("Accuracy_Estimation: {}% with {} neighbours".format(Accuracy_Estimation,num_of_neig))

elif Method =="2":
    number_of_neighbours = "false"
    Accuracy_Estimation = Cross_Validation( X, Folds, Train_indexes, All_Labels, number_of_neighbours, Method, False, False,S)
    print("Accuracy_Estimation: {}%".format(Accuracy_Estimation))


################################### TEST DATA #################################

#=============================================================================#
############################  STRATIFICATION ##################################
#=============================================================================#
Percent_dic,mini = Stratification(X,All_Labels[Train_indexes])


################################ KNN ##########################################
if Method == "1":
    for number_of_neighbours in Neighbour_list:
        num_of_neig = int(number_of_neighbours)
        Test_data = X[:,Test_indexes].T
        Train_data = X[:,Train_indexes].T
        ## DISTANCE MATRIX ##
        Dist_Matrix = np.zeros((Test_data.shape[0],Train_data.shape[0]))
        
        for  i  in range(Test_data.shape[0]):
           for j in range(Train_data.shape[0]):
               Dist_Matrix[i,j] = Dist_func(Train_data[j,:],Test_data[i,:],Distance,S)
        Matrix = Dist_Matrix.copy()
        ## ACCURACY TEST ##
        predict_labels = K_nn(Matrix,All_Labels[Train_indexes],All_Labels[Test_indexes],num_of_neig)


        Test_accuracy = Accuracy(predict_labels,Test_indexes,All_Labels)
        print("Test_accuracy: {}% with {} neighbours".format(round(Test_accuracy,2),num_of_neig))


############################ NAIVE BAEYSE #####################################
elif Method =="2":
    predict_labels = Naive(X,Test_indexes,All_Labels,Labels_list,Percentage_ci_list,Percent_dic)
    Test_accuracy = Accuracy(predict_labels,Test_indexes,All_Labels)
    print("Test_accuracy: {}%".format(round(Test_accuracy,2)))



