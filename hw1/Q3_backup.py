#!/usr/bin/python
import subprocess
import csv
import numpy as np
import scipy as sp
import random

NUM_TRAINING_ELEM=372
NUM_ITER = 1000
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
		
	for i in range(0,NUM_ITER):
		trainIn, trainOut, testIn, testOutAct=getTrainTest(X,y)
		wcap	=getWt(trainIn, trainOut)
		err 	= calcRMSErr(testIn, wcap, testOutAct)
		if err < bestFitErr:
			bestFit		= wcap
			bestFitErr 	= err
		error.append(err)	
	print "\tMean (MAE):\t", np.mean(error,dtype=np.float64)
	print "\tStd Dev (MAE):\t", np.std(error,dtype=np.float64)
	print "\n\tLeast Mean Absolute Error\t:", bestFitErr
	print "\tBest Choice of Wts (Based on Above):"
	for i in range(0,len(bestFit)):
		print "\tw%d: %s\t\t: %f" %(i, Features[i],bestFit[i])

	
	print "PART 2..."

        for i in range(0,NUM_ITER):
                trainIn, trainOut, testIn, testOutAct=getTrainTest(X,y)
                wcap    =getWt(trainIn, trainOut)
                err     = calcMAErr(testIn, wcap, testOutAct)
                if err < bestFitErr:
                        bestFit         = wcap
                        bestFitErr      = err
                error.append(err)
        print "\tMean (RMSE):\t", np.mean(error,dtype=np.float64)
        print "\tStd Dev (RMSE):\t", np.std(error,dtype=np.float64)
        print "\n\tLeast Mean Absolute Error\t:", bestFitErr
        print "\tBest Choice of Wts (Based on Above):"
        for i in range(0,len(bestFit)):
                print "\tw%d: %s\t\t: %f" %(i, Features[i],bestFit[i])
	

main()
