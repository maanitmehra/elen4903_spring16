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

def genC(n, w):
	w_cumul	= w.cumsum()
	c	= np.zeros((1,n))
	w_arr	= pd.Series(w_cumul)
	c 	= [w_arr[(w_arr > np.random.random())].index[0]+1 for i in range(0,n)]		
	return c

def genW(k):
	## We look at defining a probabiliy distribution
	## s.t. we have w[0] = (0+1)p, w[n-1] = ((n-1)+1)p
	## The sum of this must be equal to 1.
	## ==> 1p + 2p + ....n.p = 1
	p = 2.0/(k*(k+1)) 
	w = np.zeros((1,k))
	for i in range(0,k):
		w[0][i] = (i+1)*p
	return w

def gaussParam(class_k, X_train, y_train):
        ind=[]
        for i in range(0,len(y_train)):
            if y_train[i,0] == class_k:
                ind.append(i)
        x_train_cal=X_train[ind,:]
        n_y=np.size(x_train_cal,axis=0)
        mu= np.mean(x_train_cal,axis=0)
        cov=np.dot(np.transpose(x_train_cal-mu),x_train_cal)/n_y
        return mu,cov

def adaBoost():
	return 0		

	
## p1 --> Part 1
def p1():
	print "\tPart 1 >>"	
	n_list = [50, 150, 250]
	k = 4
	for n in n_list:
		w = genW(k)
		c = genC(n,w)
		#print "\t",c
		#plt.hist(c)
		plt.clf()
		z, bins, patches = plt.hist(c, 4, normed=1, facecolor='green', alpha=0.75)
#		print patches
		#sigma = np.std(c)
		#mu = np.mean(c)
		# add a 'best fit' line
		#y = mlab.normpdf( bins, mu, sigma)
		#l = plt.plot(bins, y, 'r--', linewidth=1)
		plt.plot(bins)
		#plt.imshow()
		plt.savefig(PATH+'../images/p1_hist_'+str(n)+'.jpg')
	return n_list

## part 2
def p2(X_train, y_train, X_test, y_test):
    print "\tPart 2 >>"

    train = np.hstack((X_train[:,1:],y_train))
    cl_1 = []
    cl_0 = []
    for row in train:
		if row[-1] == 1:
			cl_1.append(row[:-1])
		else:
			cl_0.append(row[:-1])
    cl_1 = np.array(cl_1)
    cl_0	= np.array(cl_0)

    mu_1 = np.mean(cl_1, axis=0)     # mu of y=1
    mu_0= np.mean(cl_0, axis=0)    # mu of y=-1

    pi_1 = len(cl_1)*1.0/(len(cl_1)+len(cl_0))
    pi_0= 1- pi1
    
    print pi1, pi0     
    
    print mu1, mu_1                
    cov = np.cov( X_train[:,1:] )
    inv_cov = np.linalg.pinv(cov)	
    

## part 3
def p3():
	print "\tPart 3 >>"


## Main Function
def main():
	print "Beginning Execution..."

	X = getData(PATH+"X.csv")
	y = getData(PATH+"y.csv")

	X_test = X[:NUM_TEST,:]
	X_train= X[NUM_TEST:,:]
	y_test = y[:NUM_TEST,:]
	y_train= y[NUM_TEST:,:]
	
	p1()
	p2(X_train, y_train, X_test, y_test)
	p3()

main()
