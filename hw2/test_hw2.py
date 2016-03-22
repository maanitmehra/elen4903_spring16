import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats.mstats import mode
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab as pl
import numpy.linalg

# Loading data
labelTest = np.genfromtxt("data/label_test.txt", dtype='int32').reshape(500,1) 
labelTrain = np.genfromtxt("data/label_train.txt", dtype='int32').reshape(5000,1)
XTest = np.genfromtxt("data/Xtest.txt", dtype='float32', delimiter =',').T
XTrain = np.genfromtxt("data/Xtrain.txt", dtype='float32', delimiter =',').T 
Q = np.genfromtxt("data/Q.txt", dtype='float32', delimiter =',')

# 3/a
labelPred = np.zeros((500,5))
# Label prediction
for i in range(0,500):
    # Calculate distance
    distance = np.sqrt(np.sum(np.power(XTrain-XTest[:,i:i+1],2), axis=0)).T.reshape(5000,1)
    # Sort distances
    labeledDist = np.concatenate((distance, labelTrain), axis=1)
    sortedDist = np.sort(labeledDist.view('float64,float64'), order=['f0'], axis=0).view(np.float)
    # Get k nearest Neighbour
    for k in range(0,5):
        neighbours = sortedDist[0:k+1,1]
        # Take majority vote
        labelPred[i,k] = mode(neighbours)[0]

# Confusion matrix and prediction accuracy
C_k = np.zeros((10,10,5)) # Confusion matrix
predAcc = np.zeros((5,1)) # Prediction accuracy
errorIdx = [] # Store indicies of misclassification
# For each case of k neighbour, iterate through every prediction
for k in range(0,5):
    for i in range(0,500):
        C_k[labelTest[i,0],labelPred[i,k],k] += 1
        # Misclassified pictures
        if labelTest[i,0] != labelPred[i,k]:
            errorIdx.append(i)
    predAcc[k,0] = np.trace(C_k[:,:,k:k+1])/500
    # Plotting 3 image for k = 1,3,5
    if k%2 == 0:
        for j in range(0,3):
            # Reconstruct image
            pic = -np.dot(Q,XTest[:,errorIdx[j+k]]).reshape(28,28)
            fig = plt.figure()            
            fig.suptitle('Misclassified kNN, k=%s, true class: %s predicted: %s, Testindex: %s'%(k+1,labelTest[errorIdx[j+k],0],int(labelPred[errorIdx[j+k],k]),errorIdx[j+k]), fontsize=18)
            plt.imshow(pic, cm.Greys_r)
            fig.show()
    errorIdx = []


# 3/b
# calculate P(y), nu and sigma
pi = np.zeros((10,1)) # P(y) for each label
nu = np.zeros((20,10)) # mean for each label
for i in range(0,5000):
    pi[labelTrain[i,0],0] += 1
    nu[:,labelTrain[i,0]] += XTrain[:,i]
pi = pi/5000    
nu = nu/500    
sigma = np.zeros((20,20,10))
# Calculate sigma
for i in range(0,5000):
    sigma[:,:,labelTrain[i,0]] += np.dot((XTrain[:,i]-nu[:,labelTrain[i,0]]).reshape(20,1),(XTrain[:,i]-nu[:,labelTrain[i,0]]).reshape(20,1).T)
sigma = sigma/500

# Prediction
f = np.zeros((10,1))
labelPredBayes = np.zeros((500,1))
for j in range(0,500):
    for i in range(0,10):
        # Applying formula from lecture P(Y=y)P(X=x|Y=y)
        f[i,0] = pi[i,0]/np.sqrt(np.linalg.det(sigma[:,:,i]))*np.exp(np.dot(np.dot((XTest[:,j]-nu[:,i]).T,np.linalg.inv(sigma[:,:,i])),(XTest[:,j]-nu[:,i]))/-2)
    # Finding the most likely label
    labelPredBayes[j,0] = np.argmax(f)    

# Confusion matrix and prediction accuracy
C_Bayes = np.zeros((10,10))
errorIdx = []
for i in range(0,500):
    C_Bayes[labelTest[i,0],labelPredBayes[i,0]] += 1
    if labelTest[i,0] != labelPredBayes[i,0]:
        errorIdx.append(i)
predAccBayes = np.trace(C_Bayes)/500

# Plotting misclassified examples
for j in range(0,3):
    # Reconstruct image
    pic = -np.dot(Q,XTest[:,errorIdx[j]]).reshape(28,28)
    fig = plt.figure()            
    fig.suptitle('Misclassified, Bayes classification, true class: %s predicted: %s, Testindex: %s'%(labelTest[errorIdx[j],0],int(labelPredBayes[errorIdx[j],0]), errorIdx[j]), fontsize=18)
    plt.imshow(pic, cm.Greys_r)
    fig.show()
    # Calculate probability distribution over the labels
    for i in range(0,10):
        f[i,0] = pi[i,0]/np.sqrt(np.linalg.det(sigma[:,:,i]))*np.exp(np.dot(np.dot((XTest[:,errorIdx[j]]-nu[:,i]).T,np.linalg.inv(sigma[:,:,i])),(XTest[:,errorIdx[j]]-nu[:,i]))/-2)
    fig = plt.figure()
    fig.suptitle('Misclassified probability distribution, Bayesion classification, Testindex: %s'%(errorIdx[j]), fontsize=18)
    # Normalize it so it will sum to 0 and be a probability distribution
    plt.bar(np.arange(len(f)),f/np.sum(f), align='center')
    fig.show()
    
# Showing mean of each gaussian
for i in range(0,10):
    # Reconstruct image
    pic = -np.dot(Q,nu[:,i]).reshape(28,28)
    fig = plt.figure()            
    fig.suptitle('Mean of Gaussian, label: %s'%(int(i)), fontsize=14)
    plt.imshow(pic, cm.Greys_r)
    fig.show()

# 3/c
nu_c = 0.1/5000
w = np.zeros((21,10))
# Augment original matrix with a row of ones
XTrain_C = np.concatenate((XTrain, np.ones((1,5000))), axis=0)
w_old = np.zeros((21,10))
likelihoodLog = np.zeros((1000,1))
for i in range(0,1000):
    # The sum in the denominator which does not depend on class
    denomSum = np.sum(np.exp(np.dot(XTrain_C.T,w_old)), axis=1, keepdims=True)
    # Calculate each w_i
    for k in range(0,10):
        w[:,k:k+1] = w_old[:,k:k+1] + nu_c*(np.sum(XTrain_C[:,k*500:(k+1)*500], axis=1,keepdims=True) -np.sum((XTrain_C.T*np.exp(np.dot(XTrain_C.T,w_old[:,k:k+1]))/denomSum).T, axis = 1, keepdims=True))
    # After each w_i is calculated update w_old for next iteration
    w_old = w
    # Calculate the loglikelihodd at each iteration
    # Term which varies over the different class
    partialsum = 0    
    for j in range(0,10):
        partialsum = partialsum + np.sum(np.dot(XTrain_C[:,j*500:(j+1)*500].T,w[:,j:j+1]))
    # Loglikelihood
    likelihoodLog[i,0] = partialsum-np.sum(np.log(np.sum(np.exp(np.dot(XTrain_C.T,w)), axis=1, keepdims=True)))
    if i%100 ==0:
        print str(i/10) + '% done.'        

# Plot likelihood
fig = plt.figure()            
fig.suptitle('Likelihood of logistic regression with %s iteration'%(int(i+1)), fontsize=18)
plt.plot(likelihoodLog)
plt.ylabel('Log likelihood')
plt.xlabel('Number of iteration')
fig.show()        
 
## Prediction
#XTest_C = np.concatenate((XTest, np.ones((1,500))), axis=0)
#predLog = np.dot(XTest_C.T, w)
#labelPredLog = np.argmax(predLog, axis=1).reshape(500,1)

# Prediction
# Augment train matrix with ones
XTest_C = np.concatenate((XTest, np.ones((1,500))), axis=0)
# Predict labels
predLog = np.exp(np.dot(XTest_C.T, w))
# Normalize
predLog = predLog/np.sum(predLog, axis=1, keepdims=True) 
labelPredLog = np.argmax(predLog, axis=1).reshape(500,1)

# Confusion matrix and prediction accuracy
C_Log = np.zeros((10,10))
errorIdx = []
for i in range(0,500):
    C_Log[labelTest[i,0],labelPredLog[i,0]] += int(1)
    if labelTest[i,0] != labelPredLog[i,0]:
        errorIdx.append(i)
predAccLog = np.trace(C_Log)/500

# Plots for misclassified image
for j in range(0,3):
    # Reconstruct image
    pic = -np.dot(Q,XTest[:,errorIdx[j]]).reshape(28,28)
    fig = plt.figure()            
    fig.suptitle('Misclassified, Log regression, true class: %s, predicted: %s, Testindex: %s'%(labelTest[errorIdx[j],0],int(labelPredLog[errorIdx[j],0]), errorIdx[j]), fontsize=18)
    plt.imshow(pic, cm.Greys_r)
    fig.show()
    # Calculate probability distribution
    predLog = np.exp(np.dot(XTest_C[:,errorIdx[j]].T, w))
    # Normalize
    predLog = predLog/np.sum(predLog)
    fig = plt.figure()
    fig.suptitle('Misclassified probability distribution, Log regression, Testindex: %s'%(errorIdx[j]), fontsize=18)
    plt.bar(np.arange(len(predLog)),predLog, align='center')
    fig.show()




