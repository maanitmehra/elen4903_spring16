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



def genC(n, w):
	w_cumul	= w.cumsum()
	c	= np.zeros((1,n))
	w_arr	= pd.Series(w_cumul)
	c 	= [w_arr[(w_arr > np.random.random())].index[0] for i in range(0,n)]		
	return c

def Adaboost(X_train, y_train, X_test, y_test, T ,classifier=None):
    n = len(X_train,axis=1)
    w = (1.0)/n * np.ones((n,1))

    for i in range(0,T):
        sample_indices = genC(n,w)
        print sample_indices
        
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
        plt.bar(np.array(range(0,len(hist))), np.array(hist), align='center')    
        plt.savefig(PATH+'../images/p1_hist_'+str(n)+'.jpg')
        
def p2():
    print "Part 2:\n"
    
def p3():
    print "Part 3:\n"
    
def main():
    print "Beginning Execution..."

    p1()
    p2()
    p3()

main()