#kilimanjaro2 submitting q2
#This is Anfield. YNWA
#201564134
import numpy as np
import scipy as sc
import math
import sys
import csv

arg_check = 0
if(sys.argv[0]=="time"):
    arg_check += 1
train_name = str(sys.argv[1+arg_check])
test_name = str(sys.argv[2+arg_check])

unproc_input = np.genfromtxt(train_name, delimiter = ',')
train_x_max = unproc_input.shape[0]
size_x_max = train_x_max

test_data = np.genfromtxt(test_name, delimiter = ',')
size_x_max = test_data.shape[0]
train_x_max *= 4
train_x_max /= 5

find_median = 0
median_cnt = 0


train_flag = unproc_input[:train_x_max,[10]]
train_pruned = unproc_input[:train_x_max,1:10]
test_pruned = test_data[:,1:10]


weight = np.zeros(9)
batch_size = 100
margin = 10
nyet = 0.05

for j in xrange(batch_size):
    for i in xrange(train_x_max):
        #print test_pruned[i].shape
        #print i
        tot = np.dot(train_pruned[i],weight.T)
        denom = np.dot(train_pruned[i],train_pruned[i].T)
        mult = (nyet * (margin - tot) / denom)
        if tot <= margin and train_flag[i] == 4 :
            weight += (mult * train_pruned[i])
        elif tot >= margin and train_flag[i] == 2:
            weight += (mult * train_pruned[i])


for i in xrange(size_x_max):
    tot = np.dot(test_pruned[i],weight.T)
    if(tot >= margin):# and test_flag[i] == 1):
        print (4)
    elif(tot < margin):# and test_flag[i] == 0):
        print (2)


weight = np.zeros(9)
batch_size = 100
margin = 10
nyet = 0.05
minima = 999999999

for j in xrange(batch_size):
    wrong=0
    for i in xrange(train_x_max):
        #print test_pruned[i].shape
        #print i
        tot = np.dot(train_pruned[i],weight.T)
        denom = np.dot(train_pruned[i],train_pruned[i].T)
        mult = (nyet * (margin - tot) / denom)
        if tot <= margin and train_flag[i] == 4 :
            weight += (mult * train_pruned[i])
            wrong +=1
        elif tot >= margin and train_flag[i] == 2:
            weight += (mult * train_pruned[i])
            wrong +=1
    if(wrong < minima):
        minima = wrong
        final = weight

weight = final

for i in xrange(size_x_max):
    tot = np.dot(test_pruned[i],weight.T)
    if(tot >= margin):# and test_flag[i] == 1):
        print (4)
    elif(tot < margin):# and test_flag[i] == 0):
        print (2)
