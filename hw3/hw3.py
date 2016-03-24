#!/usr/bin/python
import subprocess
import csv
import numpy as np
import scipy as sp
import math
import random
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

#def adaBoost():
	

	
## p1 --> Part 1
def p1():
	print "\tPart 1 >>"	
	n_list = [50, 150, 250]
	k = 4
	for n in n_list:
		w = genW(k)
		c = genC(n,w)
		print "\t",c
		#plt.hist(c)
		plt.clf()
		z, bins, patches = plt.hist(c, 4, normed=1, facecolor='green', alpha=0.75)
		plt.plot(bins)
		#plt.imshow()
		plt.savefig(PATH+'../images/p1_hist_'+str(n)+'.jpg')
	return n_list

## part 2
def p2(y_out):
	print "\tPart 2 >>"
	print y_out
	labels		= np.array(Counter(y_out[0]).keys()).astype(np.int32)
	label_counts	= np.array(Counter(y_out[0]).values()).astype(np.int32)
	label_prob	= np.array([count*1.0/np.sum(label_counts) for count in label_counts], dtype = np.float32).T.astype(np.float32)
	print label_prob
	label_arr	= np.transpose(np.vstack((labels,label_counts,label_prob)))		
	print label_arr


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
	p2(y_train.T)
	p3()
main()
