#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

EUC = 0
DIV = 1
#plt.clf()
#plt.imshow(X[:,0].reshape(32,32).T, cmap='gray')

def NMF(X, k, num_iter, penalty):
    n, m = X.shape
    W = np.random.rand(n,k)
    H = np.random.rand(k,m)    
    #W = np.ones((n,k))*1.0/(n*k)
    #H = np.ones((k,m))*1.0/(k*m)
    err_list=[]
    
    for i in range(num_iter):
        W,H,err = euclid_update(X,W,H) if penalty==EUC else div_update(X,W,H)
	print i
        err_list.append(err)
    return W,H,err_list

def div_update(X,W,H):
    W = W+1e-16
    H = H+1e-16
    temp1 = X*1.0/np.dot(W,H)    
    W_norm = W.T*1.0/np.array([np.sum(W.T, axis=1)]).T    
    H = H *  np.dot(W_norm, temp1)

    temp2 = X*1.0/np.dot(W,H)
    H_norm = H.T*1.0/np.sum(H.T,axis=0)
    W = W * np.dot(temp2, H_norm)
    
    WH = np.dot(W,H)
    err = np.sum(WH - X*np.log(WH+1e-16))
    return W,H,err
    
def euclid_update(X,W,H):
    H = H * (np.dot(W.T,X))/(np.dot(np.dot(W.T, W),H))
    W = W * (np.dot(X, H.T))/(np.dot(np.dot(W, H),H.T))
    temp = X - np.dot(W,H)
    err  = np.sum(temp**2)
    return W,H,err
    
def part1():
    print "\n\n\tPart 1:"
    k = 25  #given factorization rank
    X = np.loadtxt('./data/faces.csv', delimiter=',', dtype=int)
    num_iter = 200
    W,H,err_list=NMF(X, k, num_iter, EUC)
    
#    print "Printing Image"    
#    plt.clf()
#    plt.imshow(W[:,2].reshape(32,32).T, cmap='gray')
    
    print "Plot RMSE"
    plt.clf()
    plt.plot(err_list)
    plt.title('Min Sq. Error vs Iterations')
    plt.ylabel('$||X-WH||^2$')
    plt.xlabel('Iteration Number')
    plt.savefig('./plots/Q2_part1_err_plot.jpg') 
    plt.show()
    print err_list[0], err_list[-1]
    
    print "Printing and choosing images."
    ran_idx = np.arange(k)
    np.random.shuffle(ran_idx)
    for idx,elem in enumerate(ran_idx[:10]):
        W_name = "./plots/Q2_part1_W_%d"%elem
        x_idx = np.argmax(H[elem, :])
        X_name = "./plots/Q2_part1_W_%d_X_%d"%(elem,x_idx)
        plt.clf()
        plt.imshow(W[:,elem].reshape(32,32).T, cmap='gray')
        plt.savefig(W_name)
        plt.clf()
        plt.imshow(X[:,x_idx].reshape(32,32).T, cmap='gray')
        plt.savefig(X_name)
        
        
    
def part2():
    print "\n\n\tPart 2:"
    k = 25  #given factorization rank
    num_iter = 200
    X = np.zeros((3012,8447))
    with open("./data/nyt_data.txt") as f:
        content = f.readlines()

    content = [content.strip().split(',') for content in content]
    
    for i in range(0,len(content)):
        for j in range(0,len(content[i])-1):
            X[int(content[i][j].split(':')[0])-1,i] = int(content[i][j].split(':')[1])

    label = np.genfromtxt("./data/nytvocab.dat", delimiter='\n', dtype='str')    

    W,H,err_list=NMF(X, k, num_iter, DIV)
    W = W*1.0/np.sum(W,axis=0)
    
    print "Plot RMSE"
    plt.clf()
    plt.plot(err_list)
    plt.title('KL Divergence {D(X||WH)} vs Iterations')
    plt.ylabel('$||X.ln(\frac{1}{WH}) + WH||$')
    plt.xlabel('Iteration Number')
    plt.savefig('./plots/Q2_part2_err_plot.jpg') 
    plt.show()
    #print err_list[0], err_list[-1]

    print "Topic Modeling"
    ran_idx = np.arange(k)
    np.random.shuffle(ran_idx)
    for j,top_idx in enumerate(ran_idx[:10]):
        print "\n\n\tTopic %d\n"%top_idx        
        wrd_idx=np.argsort(W[:,top_idx].T)[::-1][:10]
        print wrd_idx
        for idx,w_idx in enumerate(wrd_idx):
            wrd = label[w_idx]
            wt  = W[w_idx, top_idx]
            print "\t\t%d\t%f\t%s"%(idx,wt,wrd)
            
def main ():
    print "Running Q2:"
    part1 ()
    part2 ()

main()
