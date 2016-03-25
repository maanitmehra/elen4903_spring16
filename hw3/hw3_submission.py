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

NUM_TEST = 183  # Define the number of test cases
T = 1000	# number of iterations of the problem

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

BINARY = 0
ONLINE = 1

def genC(n, w):
	w_cumul	= w.cumsum()
	c	= np.zeros((1,n))
	w_arr	= pd.Series(w_cumul)
	c 	= [w_arr[(w_arr > np.random.random())].index[0] for i in range(0,n)]		
	return c

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
    w = np.ones((10,1))
    return w

def Adaboost(X_train, y_train, X_test, y_test, T ,classifier):
    n = np.size(X_train,axis=0)
    pt = (1.0)/n * np.ones((n,1))
    alpha_list = []
    eps_list = []
    err_train = []
    err_test  = []
    for i in range(0,T):
        Bt = genC(n,pt)
        #print Bt
        X_bt = X_train[Bt]
        y_bt = y_train[Bt]
        if classifier ==BINARY:
            w = BinaryClassifier(X_bt, y_bt)
            print w
        
def p1():
    print "Part 1:\n"
    n_list = [50,150,250]
    w = np.array([0.1,0.2,0.3,0.4])
    for n in n_list:
        sample = genC(n,w)
        print sample
        hist, bins = np.histogram(sample, bins=len(w))
        print hist, bins
        plt.clf()
        plt.xlabel("Generated Sample Indices")
        plt.ylabel("Number")
        plt.title("Histogram for n=%d"%n)
        plt.bar(np.array(range(0,len(hist))), np.array(hist), align='center')    
        plt.savefig(PATH+'../images/p1_hist_'+str(n)+'.jpg')
        
def p2():
    print "Part 2:\n"
    Adaboost(X_train, y_train, X_test, y_test, 10 , BINARY)
    
    
def p3():
    print "Part 3:\n"
    
def main():
    print "Beginning Execution..."

    p1()
    p2()
    p3()

main()