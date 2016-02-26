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
		plt.title('$y_{pred}=%d, y_{actual}=%d$'%(y_pred[ind[i]],y_act[ind[i]]))
		filepath="./images/%s_act%d_pred%d_%d.png"%(filename,y_act[ind[i]],y_pred[ind[i]],i+1)
		plt.savefig(filepath)

def Q3_a(X_train,X_test,y_train,y_Act,tst_img_arr):
	print "Q3 part a..."
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
		print "\t\t%d\t\t\t%.2f%%"%(k+1,(np.trace(C)*100.0/num_test))
		if ((k+1)%2!=0):
			dispIneq(y_pred[:,k], y_Act, 'k_%d'%(k+1), tst_img_arr)

def Q3_b(X_train, X_test, y_train, y_Act, tst_img_arr, Q):
	print "Q3 Part b..."
	num_test = np.size(X_test,axis=0)
	tmp=Counter(y_train[:,0])	
	t_k=tmp.keys()	# holds the classes
	t_v=tmp.values()# holds the count for each class
	N=sum(t_v)	# total number of classes (N)
	pyeqY=np.zeros((1,len(t_v)))
	print "\t\t\tClass Prior"
	print "\t\tClass(Y)\tp(y=Y)"
	for i in range(0,len(t_v)):
		pyeqY[0,i]=float(t_v[i])/N
		print "\t\t%d\t\t%f"%(i,pyeqY[0,i])

	## Calculating the mean and covariance
	mu	=np.zeros((len(t_k),np.size(X_train,axis=1)))
	cov_list	=[]
	invcov_list	=[]
	for i in range(0,len(t_k)):
		mu[i,:],cov=gaussParam(t_k[i], X_train, y_train)
		cov_list.append(cov)
		invcov_list.append(np.linalg.inv(cov))
	
	y_pred          = np.zeros((np.size(X_test,axis=0), 1))
	prob_list	= []
	for i in range(0, np.size(X_test,axis=0)):
		y_pred[i,0],prob=predBayes(X_test[i,:],mu,invcov_list,cov_list,pyeqY)
		prob_list.append(prob)

	C=conMat(y_pred,y_Act)		
	accuracy = np.trace(C)*100.0/(num_test-1)
	print C
	print "Accuracy: %.2f%%" %(accuracy)
	img_arr = np.dot(Q,np.transpose(mu))
#	print np.size(img_arr,axis=0),np.size(img_arr,axis=1)
	for i in range(0,len(t_k)):
		plt.imshow(img_arr[:,i].reshape((28,28)))
                plt.title('$\mu for class=%d$'%(i))
                filepath="./images/part3b_mu_%d.png"%(i)
                plt.savefig(filepath)
	
	dispIneq(y_pred,y_Act,'Bayes_',tst_img_arr)
		
def predBayes(vec_test,mu,invcov,cov,pi):
	sig=np.zeros((1,len(invcov)))
	label_arr=np.zeros((2,len(invcov)))
	for i in range(0, len(invcov)):
		sig[0,i]=np.sqrt(np.linalg.norm(cov[i]))
		z = np.dot(vec_test-mu[i,:],np.dot(invcov[i],np.transpose(vec_test-mu[i,:])))
		label_arr[0,i] = i
		label_arr[1,i] = pi[0,i]*np.exp(-0.5*z)/sig[0,i]
	l_temp=np.transpose(label_arr)
	l_temp_fin=l_temp[l_temp[:,1].argsort()[::-1]]
	return l_temp_fin[0,0],label_arr[1,:]


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

def main():

	PATH='./support_files/mnist_csv/'

	X_train		=getData(PATH+'Xtrain.txt')
	y_train		=getData(PATH+'label_train.txt')
	Q		=getData(PATH+'Q.txt')	
	X_test		=getData(PATH+'Xtest.txt')
	y_Act		=getData(PATH+'label_test.txt')
	tst_img_arr	=np.dot(Q,np.transpose(X_test))

#	Q3_a(X_train,X_test,y_train,y_Act,tst_img_arr)
	Q3_b(X_train,X_test,y_train,y_Act,tst_img_arr,Q)
#	Q3_c	
	### DEBUGGING ###
#	dispIneq(y_Act[::-1],y_Act,"k_1",tst_img_arr)

main()
