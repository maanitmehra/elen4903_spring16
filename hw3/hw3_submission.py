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
T = 100	# number of iterations of the problem

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

def Adaboost(X_train, y_train, X_test, y_test, T ,classifier):
    n = np.size(X_train,axis=0)
    pt = (1.0)/n * np.ones((n,1))
    alpha_list = []
    eps_list = []
    err_train = []
    err_test  = []
    pt_list = []
    w_list = []
    for i in range(0,T):
        Bt = genC(n,pt)
        #print Bt
        X_bt = X_train[Bt]
        y_bt = y_train[Bt]
        if classifier ==BINARY:
            w = BinaryClassifier(X_bt, y_bt)
        elif classifier == ONLINE:
            w = OnlineClassifier(X_bt, y_bt)
        elif classifier == NONE:
            w = BinaryClassifier(X_train, y_train)
            
        w_list.append(w)
        f_t = calc_ft(X_bt, w, classifier)
        f_train = calc_ft(X_train, w, classifier)

        eps, sgn_vec = calcEps(y_train, f_train, pt)
        eps_list.append(eps)
        
        alpha = calcAlpha(eps)
        alpha_list.append(alpha)

        pt_list.append(pt)
        for idx, elem in enumerate(pt):
            pt[idx] = elem*np.exp(-alpha*sgn_vec[idx])
        pt = pt/np.sum(pt)
    print "pt_list:", pt_list    
    print "alpha_list", alpha_list
        
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
        plt.savefig(PATH+'../images/p1_hist_'+str(n)+'.jpg')
        
def p2():
    print "Part 2:\n"
    classifier = BINARY
    Adaboost(X_train, y_train, X_test, y_test, T , classifier)
    
    
def p3():
    print "Part 3:\n"
    
def main():
    print "Beginning Execution..."

    p1()
    p2()
    p3()

main()