# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 22:58:11 2016

@author: maanitm
"""

#!/usr/bin/python
import csv
import numpy as np
import pandas as pd
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm

PATH = "./cancer_csv/"
IMAGE_PATH = PATH+'../images/'
NUM_TEST = 183  # Define the number of test cases
T = 10	# number of iterations of the problem

def getData(file):
    reader=csv.reader(open(file,"rb"),delimiter=',')
    x=list(reader)
    result=np.array(x).astype(np.float32)
    return result    

X = getData(PATH+"X.csv")
y = getData(PATH+"y.csv")

X_test = X[:NUM_TEST,:]
X_train= X[NUM_TEST:,:]
y_test = y[:NUM_TEST,:]
y_train= y[NUM_TEST:,:]

NONE = 0
BINARY = 1
ONLINE = 2

def genC(n, w):
	w_cumul	= w.cumsum()
	c	= np.zeros((1,n))
	w_arr	= pd.Series(w_cumul)
	c 	= [w_arr[(w_arr > np.random.random())].index[0] for i in range(0,n)]		
	return c

def calc_ft(X_train, w, classifier):
    if classifier == BINARY or classifier == NONE:
        ft = np.sign(np.dot(X_train, w))
    elif classifier  ==  ONLINE:
	ft = np.sign(np.dot(X_train,w))

    for elem in ft:
        if not elem:
            elem = np.sign(np.random.random()-0.5)
    return ft
    
def BinaryClassifier(X_train, y_train):
    n = np.size(X_train,axis=0)
    w = np.ones((np.size(X_train,axis=1),1))

    idx_0 =[]
    idx_1 =[]    
    for idx,elem in enumerate(y_train):
        if elem == 1:
            idx_1.append(idx)
        else:
            idx_0.append(idx)
    cl_0 = X_train[idx_0,1:]
    cl_1 = X_train[idx_1,1:]

    mu_0 = np.array(np.mean(cl_0,axis=0)).reshape(1,np.size(cl_0, axis=1))
    mu_1 = np.array(np.mean(cl_1, axis=0)).reshape(1, np.size(cl_1, axis=1))

    pi_1 = len(idx_1)*1.0/n
    pi_0 = len(idx_0)*1.0/n
    
    cov  = np.cov(X_train[:,1:].T)
    #print "cov size: %d x %d"%(np.size(cov,axis=0), np.size(cov,axis=1))
    #print "mu_0 size: %d x %d"%(np.size(mu_0,axis=0),np.size(mu_0,axis=1))
    #print type(mu_0)
    w[0] = np.log(pi_1/pi_0) - 0.5*np.dot((mu_1+mu_0),np.dot(np.linalg.pinv(cov),(mu_1-mu_0).T))
    w[1:] = np.dot(np.linalg.pinv(cov),(mu_1-mu_0).T)
    #print pi_1, pi_0         
    return w

def OnlineClassifier(X_train, y_train):
    n = np.size(X_train,axis=0)
    w = np.ones((np.size(X_train,axis=1),1))
    return w

def calcEps(y_train, f_train, pt):
    sgn_vec = [np.sign(y_train[idx]*f_train[idx,0]) for idx,elem in enumerate(y_train)]
    eps = 0
    for idx,sign in enumerate(sgn_vec):
        if sign < 0:
            eps = eps + pt[idx]
    return eps, sgn_vec

def calcAlpha(eps):
    alpha = 0.5*np.log((1-eps)/eps)
    return alpha
    
def calcErr(f, y):
    C = conMat(f,y)
    num_test = len(y)
    accuracy = np.trace(C)*1.0/(num_test)
    err = 1- accuracy
    return err

def conMat(y_pred,y_act):
        # C--> ConMatrix
        C=np.zeros((10,10))
        for i in range(0, np.size(y_pred)):
                real=int(y_act[i])
                pred=int(y_pred[i])
                C[real,pred]=C[real,pred]+1
        return C
    
    
def Adaboost(X_train, y_train, X_test, y_test, T ,classifier, name):
    n = np.size(X_train,axis=0)
    pt = (1.0)/n * np.ones((n,1))
    
    #alpha & epsilon: parameters used to update the values.
    alpha_list = []
    eps_list = []
    
    #error vectors
    err_train_list = []
    err_test_list  = []

    #pt as described in the documentation
    pt_list = []
    w_list = []
    
    f_boost_te = 0
    f_boost_tr = 0
    for i in range(0,T):
        Bt = genC(n,pt)
        #print Bt
        X_bt = X_train[Bt]
        y_bt = y_train[Bt]
        if classifier ==BINARY:
            classifier_name = "Binary_Classifier"
            w = BinaryClassifier(X_bt, y_bt)
        elif classifier == ONLINE:
            classifier_name = "Online_Classifier"
            w = OnlineClassifier(X_bt, y_bt)
        elif classifier == NONE:
            w = BinaryClassifier(X_train, y_train)
            
        w_list.append(w)
        f_train = calc_ft(X_train, w, classifier)
        f_test = calc_ft(X_test, w, classifier)

        eps, sgn_vec = calcEps(y_train, f_train, pt)
        eps_list.append(eps)
        
        alpha = calcAlpha(eps)
        alpha_list.append(alpha)

        pt_list.append(pt)
        for idx, elem in enumerate(pt):
            pt[idx] = elem*np.exp(-alpha*sgn_vec[idx])
        pt = pt/np.sum(pt)
        
        f_boost_te = f_boost_te + (alpha*f_test)
        f_boost_test = np.sign(f_boost_te)
        
        err_test = calcErr(f_boost_test, y_test)
        err_test_list.append(err_test)

        f_boost_tr = f_boost_tr + (alpha*f_train)
        f_boost_train = np.sign(f_boost_tr)
        
        err_train = calcErr(f_boost_train, y_train)
        err_train_list.append(err_train)
        

#    f_boost = np.sum(f_boost)    
    print "pt_list:", pt_list    
    print "alpha_list", alpha_list
    print "training Error:", err_train_list
    print "testing Error:", err_test_list
    plt.clf()
    plt.plot(err_train_list, 'r-', label="Training error")
    plt.plot(err_test_list, 'b-', label="Testing Error")    
    plt.xlabel("Values of t")
    plt.ylabel("Error Values")
    plt.title("Error v/s iteration plot for "+classifier_name)
    plt.legend(loc='upper right', shadow=True)
    plt.savefig(name)
    
def p1():
    print "Part 1:\n"
    n_list = [50,150,250]
    w = np.array([0.1,0.2,0.3,0.4])
    for n in n_list:
        sample = genC(n,w)
        #print sample
        hist, bins = np.histogram(sample, bins=len(w))
        #print hist, bins
        plt.clf()
        plt.xlabel("Generated Sample Indices")
        plt.ylabel("Number")
        plt.title("Histogram for n=%d"%n)
        plt.bar(np.array(range(0,len(hist))), np.array(hist), align='center')    
        plt.savefig(IMAGE_PATH+'p1_hist_'+str(n)+'.jpg')
        
def p2():
    print "Part 2:\n"
    classifier = BINARY
    Adaboost(X_train, y_train, X_test, y_test, T , classifier, IMAGE_PATH+"binary_classifier_error.jpg")
    
    
def p3():
    print "Part 3:\n"
    classifier = ONLINE
    Adaboost(X_train, y_train, X_test, y_test, T , classifier, IMAGE_PATH+"p3_error.jpg")
    
def main():
    print "Beginning Execution..."

    p1()
    p2()
    p3()

main()
