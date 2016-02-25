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

## euclidVect function returns euclidean distances
## between an input test vector & all the (x1...xn)in 
## the  X_train.
##
## This function calls the euclidDist function.
def euclidVect(X_train, y_train, vec_test):
	euclid_arr=np.hstack((y_train,y_train))
	for i in range(np.size(X_train,axis=0)):
		euclid_arr[i,0]=euclidDist(vec_test,X_train[i,:])	
	return euclid_arr

## This function takes in training data (X_in, y_in),
## a sample test vector and a desired 'K' value.
##
## It returns a vector with predicted y values
## for each k value upto the K_max value input.
def knn_main(K_max,X_train, y_train, vec_test):

	knn_vec = euclidVect(X_train,y_train,vec_test)

	## We look to sort the knn_vec with the help of
	## first column.
	## Source: http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
	knn_vec_sorted=knn_vec[knn_vec[:,0].argsort()]	
	y_pred=np.zeros((1,K_max))

	for k in range(1, K_max+1):

		y_out		=knn_vec_sorted[0:k,-1]	
		labels		=Counter(y_out).keys()
		label_counts	=Counter(y_out).values()
	
		label_arr	=np.transpose(np.vstack((labels,label_counts)))	
	
		label_arr_sorted=label_arr[label_arr[:,1].argsort()[::-1]]
		label		=label_arr_sorted[0,0]	
		y_pred[0,k-1]	=label
#		print label
	return y_pred

def Q3_a(X_train,X_test,y_train,y_Act,Q):
	K_max 		= MAX_K_VAL
	num_test 	= np.size(X_test,axis=0)#number of test vectors
	# y_pred stores the final predicted values,
	# each column corresponds to one k value.
	# each row corresponds to a pt prediction
	y_pred		= np.zeros((num_test, K_max))
	for i in range(num_test):
		y_pred[i,:]	= knn_main(K_max, X_train, y_train, X_test[i,:])
	print np.hstack((y_pred,y_Act))

def main():

	PATH='./support_files/mnist_csv/'

	X_train	=getData(PATH+'Xtrain.txt')
	y_train	=getData(PATH+'label_train.txt')
	Q	=getData(PATH+'Q.txt')	
	X_test	=getData(PATH+'Xtest.txt')
	y_Act	=getData(PATH+'label_test.txt')

	print np.size(X_test,axis=0)
	print np.size(y_Act,axis=0)
	print "Q3 part a..."
	Q3_a(X_train,X_test,y_train,y_Act,Q)
	## y_pred: Predicted Array of Labels
#	y_pred	=Q3_a(X_train,y_train,X_test,y_Act)	

	#### TESTING IN MAIN FUNCTION
#	vec2=knn_main(X_train,y_train,X_test[-1,:])
#	vec=euclidVect(X_train,y_train,X_test[-1,:])
#	a= vec[vec[:,0].argsort()]
#	print a
#	k=2
#	zz= a[0:k,-1]
#	print zz
#	Counter(zz)
#       labels          =Counter(zz).keys()
#	label_counts    =Counter(zz).values()
#      	label_arr       =np.transpose(np.vstack((labels,label_counts)))
#	print "label_arr:\t",label_arr
#      	label_arr_sorted=label_arr[label_arr[:,1].argsort()[::-1]]
#	print "label_arr_after_sorting\t", label_arr_sorted
#      	label           =label_arr_sorted[0,0]
#	print label
#	print "vec from function:", vec2
main()
