#!/usr/bin/python
import subprocess
import csv
import numpy as np
import scipy as sp
import math
import random


NUM_TRAINING_ELEM=372
NUM_ITER = 1000
NUM_TESTING_ELEM=20

MAE=1
RMSE=2

PRINT=1
NO_PRINT=0

Features =['intercept term',
	'no. of cylinders',
        'displacement',
        'horsepower\t',
        'weight\t',
        'acceleration',
        'model year\t']
OutputUnits =['miles per gallon']

def getData(file):
	reader=csv.reader(open(file,"rb"),delimiter=',')
	x=list(reader)
	result=np.array(x).astype(np.float32)
	return result

def getText(file):
        reader=csv.reader(open(file,"rb"),delimiter=',')
        x=list(reader)
        result=np.array(x)
        return result

def getWt(X,y):
	Xt	=np.transpose(X)
	XtXinv	=np.linalg.inv(np.dot(Xt,X))
	Xty	=np.dot(Xt,y)
	w 	=np.dot(XtXinv,Xty)
	return w 

def calcOp(X,w):
	y=np.dot(X,w)
	return y

def calcRMSErr(X,w,y):
	testOutAct = y
	testOutCalc = calcOp(X,w)
	error = np.std(testOutAct-testOutCalc,dtype=np.float64)
	return error

def calcMAErr(X,w,y):
	testOutAct = y
        testOutCalc = calcOp(X,w)
        error = np.mean(np.abs(testOutAct-testOutCalc),dtype=np.float64)
        return error

def calcRAWErr(X,w,y):
        testOutAct = y
        testOutCalc = calcOp(X,w)
        error = testOutAct-testOutCalc
	if np.size(error,axis=0)!=NUM_TESTING_ELEM: 
		print np.size(error,axis=0), "x", np.size(error,axis=1)
        return error

def getTrainTest(X,y):
	rowLength	=np.size(X,axis=0)
	colLength	=np.size(X,axis=1)
	testRow		=[]
	trainRow	=np.sort(random.sample(range(0,rowLength),
	NUM_TRAINING_ELEM))
	for i in range(0, rowLength):
		if i not in trainRow:
			testRow.append(i)

	trainIn		=X[trainRow,:]
	testIn		=X[testRow, :]
	trainOut	=y[trainRow,:]
	testOutAct	=y[testRow, :]

	return trainIn, trainOut, testIn, testOutAct

def mat_pow(X,p):
        rows = np.size(X,axis=0)
        cols = np.size(X,axis=1)	
	X_mod=np.zeros((1,cols))
	for row in X:
		X_mod=np.vstack((X_mod,np.power(row,p)))
        return X_mod[1:,:]

def mean_std_calc(X,y,err_type, should_print):
        Error_Name	="MAE"
	error 		= []
	raw_diff	= np.zeros((NUM_TESTING_ELEM,1))
	bestFitErr	= np.inf
        for i in range(0,NUM_ITER):
                trainIn, trainOut, testIn, testOutAct=getTrainTest(X,y)
                wcap    =getWt(trainIn, trainOut)
                if err_type == RMSE:
                        err     = calcRMSErr(testIn, wcap, testOutAct)
                        Error_Name="RMSE"
                else:
                        err     = calcMAErr(testIn,wcap,testOutAct)
                if err < bestFitErr:
                        bestFit         = wcap
                        bestFitErr      = err
                error.append(err)
		raw=calcRAWErr(testIn,wcap,testOutAct)
		raw_diff = np.hstack((raw_diff, raw))
	mean_val = np.mean(error,dtype=np.float64)
	std_val  = np.std(error,dtype=np.float64)
	if should_print:
	        print "\tMean (",Error_Name,"):\t", mean_val
        	print "\tStd Dev (",Error_Name,"):\t", std_val
        	print "\n\tLeast Mean Absolute Error\t:", bestFitErr
        	print "\tBest Choice of Wts (Based on Above):"
        	for i in range(0,len(bestFit)):
                	print "\tw%d: %s\t\t: %f" %(i, Features[i],bestFit[i])
	return mean_val,std_val, raw_diff[:,1:]

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
mpl.rcParams['savefig.dpi'] = 100
import matplotlib.mlab as mlab


def plot_hist(x, filename):
	val=filename[1]
	filename=str(filename+".png")
	plt.clf()
	# the histogram of the data
	n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
	sigma = np.std(x)
	mu = np.mean(x)
	# add a 'best fit' line
	y = mlab.normpdf( bins, mu, sigma)
	l = plt.plot(bins, y, 'r--', linewidth=1)

	plt.xlabel('$y^{test} - y^{pred}$')
	plt.ylabel('Probability')
	plt.title(r'$\mathrm{For\ p='+val+'\ Histogram\ of\ errors:}\ \mu='+str(mu.astype(np.float32))+',\ \sigma='+str(sigma.astype(np.float32))+'$')
	plt.legend(('Gaussian Fit','Histogram'),loc='upper right')
	plt.grid(True)

	plt.savefig(filename)


def log_likelihood(data):
    n = len(data)
    mu = calcMu(data)
    var = calcVar(data)
    ll = -(n/2.) * math.log(2 * math.pi)-(n/2.) * math.log(var)-(1 / (2. * var)) * sum([(x - mu) ** 2 for x in data])
    return ll

### Maximum Likelihood Estimators
def calcMu(data):
    mu = sum(data) / float(len(data))
    return mu

def calcVar(data):
    mu = calcMu(data)
    var = sum([(x - mu) ** 2 for x in data]) / float(len(data)-1) 
    return var



def main():
	X=getData("./data_csv/X.txt")
        y=getData("./data_csv/y.txt")

	print "PART 1..."

	print "Sub part a. "
	trainIn		=X[0:NUM_TRAINING_ELEM,:]
	trainOut	=y[0:NUM_TRAINING_ELEM,:]
	testIn		=X[NUM_TRAINING_ELEM:,:]	
	testOutAct	=y[NUM_TRAINING_ELEM:,:]
	wcap		=getWt(trainIn,trainOut)
	testOutCalc	=calcOp(testIn,wcap)
	err		=calcMAErr(testIn,wcap,testOutAct)
	print "\tMAE for first %d elements: %f" %(NUM_TRAINING_ELEM,err)	
	print "w_ml for the above:"
        for i in range(0,len(wcap)):
            print "\tw%d: %s\t\t: %f" %(i, Features[i],wcap[i])
	    if wcap[i]>0:
		print "\t %s Increase With Increasing, %s"%(OutputUnits[0], Features[i])
	    elif wcap[i]<0:
		print "\t %s Decrease With Increasing, %s"%(OutputUnits[0], Features[i])
	    else:
		print "\t %s Does not Change irrespective, of %s" %(OutputUnits[0], Features[i])

	print "Sub part b."
	bestFit = wcap
	bestFitErr = err
	error =[]
	
	mean_std_calc(X,y,MAE, PRINT)

	print "PART 2..."

	p = []
	err = []
	std = []
	X_mod = X
	ll_max = -np.inf
	e_min = np.inf
	e_idx = 0 
	ll_idx= 0
	print "\tp\terr\t\tstd\t\tLog Likelihood"
	for i in range(1, 6):
		p.append(i)
		e,s,diff_vec= mean_std_calc(X_mod,y,RMSE, 0)
		X_mod = np.hstack((X_mod, mat_pow(X[:,1:],i+1)))
		err.append(e)
		std.append(s)
		plot_hist(diff_vec.reshape(np.size(diff_vec, axis=0)*np.size(diff_vec, axis=1),1),"p"+str(i))
		ll= log_likelihood(diff_vec.reshape(np.size(diff_vec, axis=0)*np.size(diff_vec, axis=1),1))
		print "\t%d\t%f\t%f\t%f" %(i,e,s,ll) 
		if e < e_min:
			e_min = e
			e_idx = i
		if ll > ll_max:
			ll_max = ll
			ll_idx = i			

	print "\n\tp=", e_idx, ": best as per the min err values"
	print "\tp=", ll_idx, ": best as per the log likelihood values"

main()
