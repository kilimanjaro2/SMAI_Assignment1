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
#test_name = str(sys.argv[2+arg_check])

unproc_input = np.genfromtxt(train_name, delimiter = ',')
train_x_max = unproc_input.shape[0]
size_x_max = train_x_max

train_x_max *= 4
train_x_max /= 5

find_median = 0
median_cnt = 0

learning_rate = 0.3

train_flag = unproc_input[:train_x_max,[10]]
test_flag = unproc_input[train_x_max:,[10]]
train_pruned = unproc_input[:train_x_max,1:10]
test_pruned = unproc_input[train_x_max:,1:10]


for i in xrange(train_x_max):
    if not(math.isnan(train_pruned[i][5])):
        find_median += train_pruned[i][5]
        median_cnt += 1

median_fin = find_median / median_cnt

for i in xrange(train_x_max):
    if (math.isnan(train_pruned[i][5])):
        train_pruned[i][5] = median_fin

for i in xrange(train_x_max,size_x_max):
    if (math.isnan(test_pruned[i-train_x_max][5])):
        test_pruned[i-train_x_max][5] = median_fin

weight = np.zeros(9)
batch_size = 10
margin = 10

for i in xrange(train_x_max):
    tot = np.dot(test_pruned[i],weight.T)
    if tot >= 0 and train_flag == 2 :

    elif tot < 0 and train_flag == 4:
        #do something else
