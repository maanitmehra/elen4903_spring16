#!/isr/bin/env python

#import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg as sp_linalg

X = np.loadtxt('./data/cfb2015scores.csv', delimiter=',', dtype = int)
L = np.loadtxt('./data/legend.txt', delimiter='\n', dtype=str)

num_iter = 2500
NUM_TEAMS = np.max(X[:,[0,2]])
#Changing Indices to Zero Indexed Numbers
X[:, [0,2]]=X[:, [0,2]]-1

M = np.zeros((NUM_TEAMS, NUM_TEAMS))

## Preparing M
for scores in X:
    win = 0 if scores[1] >= scores[3] else 2
    los = 2 if scores[1] >= scores[3] else 0
    win_idx = scores[win]
    los_idx = scores[los]    
    win_score = scores[win+1]
    los_score = scores[los+1]
    tot = win_score + los_score
    
    M[los_idx, win_idx] = M[los_idx, win_idx] + 1 + (win_score*1.0/tot) 
    M[win_idx, win_idx] = M[win_idx, win_idx] + 1 + (win_score*1.0/tot) 
    M[los_idx, los_idx] = M[los_idx, los_idx] + (los_score*1.0/tot) 
    M[win_idx, los_idx] = M[win_idx, los_idx] + (los_score*1.0/tot)

M_norm = np.array([row*1.0/np.sum(row) for row in M]) 
vals, vecs = sp_linalg.eigs(M_norm.T, k=1)
w_inf = vecs.T[0]/np.sum(vecs.T[0])
w_0 = np.ones((1,NUM_TEAMS))[0]/NUM_TEAMS
w_list = []
diff = []
w_t = w_0
for i in range(num_iter):
        w_t= np.dot(w_t,M_norm)
        w_list.append(w_t)        
        diff.append( np.sum( np.abs(w_t - w_inf) ) )

print "Q1:"
t=[10, 100, 1000, 2500]

for elem in t:
    print "\n\t For t = %d:"%elem
    print "\n\t\tRank\tw_t\t\tCollege"
    w = w_list[elem-1]
    top_25 = np.argsort(w)[::-1][:25]
    for idx,rank in enumerate(top_25):
        print "\t\t%d\t%f\t%s"%(idx+1,w[rank],L[rank])

print "\n\n\t|| w_2500 - w_inf || = %f" %diff[-1]

plt.clf()
plt.ylabel('$|| w_{t} - w_{\infty} || _{1}$')
plt.xlabel('State Update Iteration (t)')
plt.title('$|| w_{t} - w_{\infty} || _{1}$'+' with State updates')
plt.plot(diff)
plt.savefig('./Q1_plot.jpg')
plt.show()
