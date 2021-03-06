#kilimanjaro2 submitting q1
#This is Anfield. YNWA
#201564134

#Pre Processing

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

train = np.genfromtxt(train_name, delimiter = ',')
test = np.genfromtxt(test_name, delimiter = ',')
train_x_max = train.shape[0]
test_x_max = test.shape[0]

train_flag = train[:,[0]]
#test_flag = test[:,[0]]
train_pruned = train[:,1:]
test_pruned = test[:,:]

#single perceptron witout margin

weight = np.zeros(784)

for i in xrange(train_x_max):
    tot = np.dot(train_pruned[i],weight.T)
    if (tot >= 0 and train_flag[i] == 0):
        weight -= train_pruned[i]
    elif (tot < 0 and train_flag[i] == 1):
        weight += train_pruned[i]

for i in xrange(test_x_max):
    tot = np.dot(test_pruned[i],weight.T)
    if(tot >= 0):# and test_flag[i] == 1):
        print (1)
    elif(tot < 0):# and test_flag[i] == 0):
        print (0)


#batch perceptron without margin
weight = np.zeros(784)
dump = np.zeros(784)
batch_size = 100

while(batch_size > 0):
    dump = np.zeros(784)
    for i in xrange(train_x_max):
        tot = np.dot(train_pruned[i],weight.T)
        if (tot >= 0 and train_flag[i] == 0):
            dump -= train_pruned[i]
        elif (tot < 0 and train_flag[i] == 1):
            dump += train_pruned[i]
    weight += dump
    batch_size -= 1

for i in xrange(test_x_max):
    tot = np.dot(test_pruned[i],weight.T)
    if(tot >= 0):
        print (1)
    elif(tot < 0):
        print (0)

#single perceptron with margin

margin = 100
for i in xrange(train_x_max):
    tot = np.dot(train_pruned[i],weight.T)
    if (tot >= margin and train_flag[i] == 0):
        weight -= train_pruned[i]
    elif (tot <= margin and train_flag[i] == 1):
        weight += train_pruned[i]

for i in xrange(test_x_max):
    tot = np.dot(test_pruned[i],weight.T)
    if(tot >= 0):
        print (1)
    elif(tot < 0):
        print (0)

#batch perceptron with margin
weight = np.zeros(784)
dump = np.zeros(784)
batch_size = 50
margin = 10

while(batch_size > 0):
    dump = np.zeros(784)
    for i in xrange(train_x_max):
        tot = np.dot(train_pruned[i],weight.T)
        if (tot >= margin and train_flag[i] == 0):
            dump -= train_pruned[i]
        elif (tot <= margin and train_flag[i] == 1):
            dump += train_pruned[i]
    weight += dump
    batch_size -= 1

for i in xrange(test_x_max):
    tot = np.dot(test_pruned[i],weight.T)
    if(tot >= 0):
        print (1)
    elif(tot < 0):
        print (0)
    #print cnt
