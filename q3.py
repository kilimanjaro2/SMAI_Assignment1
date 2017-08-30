import math
import pandas as pd
import sys
from pandas import DataFrame
from collections import Counter
import numpy as np


def generate_input(df):
    manipulate(df)
    typecast(df)

def manipulate(df):
    df.loc[:,'satisfaction_level'] *= 100
    df.loc[:,'satisfaction_level'] /= 10
    df.loc[:,'last_evaluation'] *= 100
    df.loc[:,'last_evaluation'] /= 10
    df.loc[:,'average_montly_hours'] /= 25

def typecast(df):
    df.loc[:,'satisfaction_level'] = df.loc[:,'satisfaction_level'].astype(int)
    df.loc[:,'last_evaluation'] = df.loc[:,'last_evaluation'].astype(int)
    df.loc[:,'average_montly_hours'] = df.loc[:,'average_montly_hours'].astype(int)


def net_entropy(a_list):
    cnt = Counter(x for x in a_list)
    num_instances = len(a_list)
    probs = [x /(num_instances * 1.0) for x in cnt.values()]
    total = 0.0
    for prob in probs:
        if prob != 0:
            total += -prob*math.log(prob, 2)
    return total

def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    df_split = df.groupby(split_attribute_name)

    # Calculate Entropy for Target Attribute, as well as Proportion of Obs in Each Data-Split
    #nobs = len(df.index) * 1.0
    df_agg_ent = df_split.agg({target_attribute_name : [net_entropy, lambda x: len(x)/float(len(df.index) * 1.0)] })[target_attribute_name]
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    # Calculate Information Gain:
    return net_entropy(df[target_attribute_name]) - sum( df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )

def build_tree(df, target_attribute_name, attribute_names, default_class=None):
    cnt = Counter(x for x in df[target_attribute_name])
    if len(cnt) == 1:
        return cnt.keys()[0]

    if df.empty or (not attribute_names):
        return default_class

    index_of_max = cnt.values().index(max(cnt.values()))
    default_class = cnt.keys()[index_of_max]
    gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names]
    #index_of_max = gainz.index(max(gainz))
    best_attr = attribute_names[gainz.index(max(gainz))]
    tree = {best_attr:{}}
    remaining_attribute_names = [i for i in attribute_names if i != best_attr]

    for attr_val, data_subset in df.groupby(best_attr):
        subtree = build_tree(data_subset,
                    target_attribute_name,
                    remaining_attribute_names,
                    default_class)
        tree[best_attr][attr_val] = subtree
    return tree

def classify(instance, tree, default=None):
    attribute = tree.keys()[0]
    if instance[attribute] in tree[attribute].keys():
        if isinstance(tree[attribute][instance[attribute]], dict):
            x = classify(instance, tree[attribute][instance[attribute]])
            return x
        else:
            return tree[attribute][instance[attribute]]
    else:
        return default

if __name__ == '__main__':

    #Loading data from data_frame
    train_data = DataFrame.from_csv(sys.argv[1], index_col=None)
    test = DataFrame.from_csv(sys.argv[2], index_col=None)

    attribute_names = list(train_data.columns)
    attribute_names.remove('left')

    training_data = train_data
    test_data  = test

    generate_input(training_data)
    generate_input(test_data)

    #Create Training Tree
    train_tree = build_tree(training_data, 'left', attribute_names)
    #Creating test_data
    test_data['prediction'] = test_data.apply(classify, axis=1, args=(train_tree,0) )
    #Final Test_data
    test_data.fillna(0, inplace=True)

    for i in test_data['prediction']:
        op = int(i)
        #print op

    print 'Accuracy is ' + str( sum(test_data['left']==test_data['prediction'] ) / (1.0*len(test_data.index)) )
