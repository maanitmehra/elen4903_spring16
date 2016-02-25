import csv
import numpy as np
import math
from collections import Counter
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


MAX_K_VAL=9
NUM_IMAGES=3
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
	return y_pred

def conMat(y_pred,y_act):
	# C--> ConMatrix
	C=np.zeros((10,10))
	for i in range(0, np.size(y_pred)):
		real=int(y_act[i])
		pred=int(y_pred[i])
		C[real,pred]=C[real,pred]+1
	return C

def dispIneq(y_pred,y_act, filename, tst_img_arr):
	ind=[]
	for i in range(0,len(y_pred)):
		if y_pred[i] != y_act[i]:
			ind.append(i) 
	for i in range(0,NUM_IMAGES):
		arr=tst_img_arr[:,ind[i]]
		plt.imshow(arr.reshape((28,28)))
		filepath="./images/%s_act%d_pred%d_%d.png"%(filename,y_act[ind[i]],y_pred[ind[i]],i+1)
		plt.savefig(filepath)

def Q3_a(X_train,X_test,y_train,y_Act,tst_img_arr):
	K_max 		= MAX_K_VAL
	num_test 	= np.size(X_test,axis=0)#number of test vectors
	# y_pred stores the final predicted values,
	# each column corresponds to one k value.
	# each row corresponds to a pt prediction
	y_pred		= np.zeros((num_test, K_max))
	for i in range(num_test):
		y_pred[i,:]	= knn_main(K_max, X_train, y_train, X_test[i,:])
		if ((i*10)%num_test==0):
			print "\t",i*100.0/num_test , "% loaded"

#	print np.hstack((y_pred,y_Act))
	print "\t\tk\t\tPrediction Accuracy"
	for k in range(0,K_max):
#		print "\tConfusion Matrix for k=%d:"%(k+1)
		C = conMat(y_pred[:,k],y_Act)
		print "\t\t%d\t\t\t%.2f%%"%(k+1,(np.trace(C)*100/500.0))
		if ((k+1)%2!=0):
			dispIneq(y_pred[:,k], y_Act, 'k_%d'%(k+1), tst_img_arr)

def main():

	PATH='./support_files/mnist_csv/'

	X_train		=getData(PATH+'Xtrain.txt')
	y_train		=getData(PATH+'label_train.txt')
	Q		=getData(PATH+'Q.txt')	
	X_test		=getData(PATH+'Xtest.txt')
	y_Act		=getData(PATH+'label_test.txt')
	tst_img_arr	=np.dot(Q,np.transpose(X_test))

	print "Q3 part a..."
	Q3_a(X_train,X_test,y_train,y_Act,tst_img_arr)
	
	### DEBUGGING ###
#	dispIneq(y_Act[::-1],y_Act,"k_1",tst_img_arr)

main()
