import csv
import numpy as np
import math
from collections import Counter

MAX_K_VAL=5

## Function to read Data from the file
def getData(file):
        reader=csv.reader(open(file,"rb"),delimiter=',')
        x=list(reader)
        result=np.array(x).astype(np.float32)
        return result

## For two 1D arrays, calc Euclidean Distance
def euclidDist(vec1,vec2):
	dist_vec= vec1-vec2
	dist_vec= dist_vec**2
	dist	=np.sqrt(sum(dist_vec))
	return dist

## Function to implement the k-NN
## Input  : Training Data(inputs & output)
##        : Test Input
## Outputs: Predicted Class
def knn(k, X_train, y_train, vec_test):
	iter=np.size(X_train,axis=0)
	edd_vec=np.empty(y_train.shape)	# Euclid Dist Vector
	for i in range(0,iter):
		edd_vec[i]=euclidDist(X_train[i,:],vec_test)
	## knn_vec stores an appended distance and
	## trained value vector. 
	## We sort the first col, and look to find
	## matching labels.
	knn_vec=np.hstack((edd_vec,y_train))	

	## We look to sort the knn_vec with the help of
	## first column.
	## Source: http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
	knn_vec_sorted=knn_vec[knn_vec[:,1].argsort()]	
	
	y_out		=knn_vec_sorted[0:k,-1]
	labels		=Counter(y_out).keys()
	label_counts	=Counter(y_out).values()
	
	label_arr	=np.transpose(np.vstack((labels,label_counts)))	
	
	label_arr_sorted=label_arr[label_arr[:,1].argsort()[::-1]]
	label		=label_arr_sorted[0,0]	
	return label

## Q3 part a main function
def Q3_a(X_train,y_train,X_test,y_act):
	rows		=np.size(X_test,axis=0)
	y_pred		=[]
	for k_val in range(1,MAX_K_VAL+1):
		y_pred_ind	=[]
		for i in range(0,rows):
			y_pred_ind.append(knn(k_val,X_train,y_train,X_test[i,:]))

		y_pred=np.hstack((y_pred,y_pred_ind))
		print "K=%d completed"%k_val
	return y_pred

def main():

	PATH='./support_files/mnist_csv/'

	X_train	=getData(PATH+'Xtrain.txt')
	y_train	=getData(PATH+'label_train.txt')
	Q	=getData(PATH+'Q.txt')	
	X_test	=getData(PATH+'Xtest.txt')
	y_Act	=getData(PATH+'label_test.txt')
#	row=np.size(X_train, axis=0)
#	col=np.size(X_train, axis=1)
#	print row,col	
#	print np.size(y_train)

	print "Q3 part a..."
	## y_pred: Predicted Array of Labels
	y_pred	=Q3_a(X_train,y_train,X_test,y_Act)	
	print y_pred
	print y_Act
#main()

def testEuclid():
	a=np.random.randint(1,4,4)
	b=np.random.randint(1,4,4)
	
	dist=euclidDist(a,b)
	print "a:\t",a
	print "b:\t",b
	print "dist:\t",dist

testEuclid()
