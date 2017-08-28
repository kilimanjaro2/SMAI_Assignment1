import math
import pandas as pd
import sys
from pandas import DataFrame
from collections import Counter
from pprint import pprint

df_shroom = DataFrame.from_csv(sys.argv[1])

def entropy(probs):
    '''
    Takes a list of probabilities and calculates their entropy
    '''
    return sum( [-prob*math.log(prob, 2) for prob in probs] )


def entropy_of_list(a_list):
    '''
    Takes a list of items with discrete values (e.g., poisonous, edible)
    and returns the entropy for those items.
    '''
    # Tally Up:
    cnt = Counter(x for x in a_list)

    # Convert to Proportion
    num_instances = len(a_list)*1.0
    probs = [x / num_instances for x in cnt.values()]

    # Calculate Entropy:
    return entropy(probs)

# The initial entropy of the poisonous/not attribute for our dataset.
#total_entropy = entropy_of_list(df_shroom['left'])
#print total_entropy

def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    '''
    Takes a DataFrame of attributes, and quantifies the entropy of a target
    attribute after performing a split along the values of another attribute.
    '''

    # Split Data by Possible Vals of Attribute:
    df_split = df.groupby(split_attribute_name)

    # Calculate Entropy for Target Attribute, as well as Proportion of Obs in Each Data-Split
    nobs = len(df.index) * 1.0
    df_agg_ent = df_split.agg({target_attribute_name : [entropy_of_list, lambda x: len(x)/nobs] })[target_attribute_name]
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    if trace: # helps understand what fxn is doing:
        print df_agg_ent

    # Calculate Information Gain:
    new_entropy = sum( df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )
    old_entropy = entropy_of_list(df[target_attribute_name])
    return old_entropy-new_entropy

#print '\nExample: Info-gain for best attribute is ' + str( information_gain(df_shroom, 'odor', 'class') )

def id3(df, target_attribute_name, attribute_names, default_class=None):

    ## Tally target attribute:
    from collections import Counter
    cnt = Counter(x for x in df[target_attribute_name])

    ## First check: Is this split of the dataset homogeneous?
    # (e.g., all mushrooms in this set are poisonous)
    # if yes, return that homogenous label (e.g., 'poisonous')
    if len(cnt) == 1:
        return cnt.keys()[0]

    ## Second check: Is this split of the dataset empty?
    # if yes, return a default value
    elif df.empty or (not attribute_names):
        return default_class

    ## Otherwise: This dataset is ready to be divvied up!
    else:
        # Get Default Value for next recursive call of this function:
        index_of_max = cnt.values().index(max(cnt.values()))
        default_class = cnt.keys()[index_of_max] # most common value of target attribute in dataset

        # Choose Best Attribute to split on:
        gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names]
        index_of_max = gainz.index(max(gainz))
        best_attr = attribute_names[index_of_max]

        # Create an empty tree, to be populated in a moment
        tree = {best_attr:{}}
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]

        # Split dataset
        # On each split, recursively call this algorithm.
        # populate the empty tree with subtrees, which
        # are the result of the recursive call
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset,
                        target_attribute_name,
                        remaining_attribute_names,
                        default_class)
            tree[best_attr][attr_val] = subtree
        return tree

# Get Predictor Names (all but 'class')
attribute_names = list(df_shroom.columns)
attribute_names.remove('left')

# Run Algorithm:

tree = id3(df_shroom, 'left', attribute_names)
#pprint(tree)

def classify(instance, tree, default=None):
    attribute = tree.keys()[0]
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict): # this is a tree, delve deeper
            return classify(instance, result)
        else:
            return result # this is a label
    else:
        return default

df_shroom['predicted'] = df_shroom.apply(classify, axis=1, args=(tree,'0') )
    # classify func allows for a default arg: when tree doesn't have answer for a particular
    # combitation of attribute-values, we can use 'poisonous' as the default guess (better safe than sorry!)

print 'Accuracy is ' + str( sum(df_shroom['left']==df_shroom['predicted'] ) / (1.0*len(df_shroom.index)) )

training_data = df_shroom.iloc[1:-1000] # all but last thousand instances
test_data  = df_shroom.iloc[-1000:] # just the last thousand
train_tree = id3(training_data, 'left', attribute_names)

test_data['predicted2'] = test_data.apply(                                # <---- test_data source
                                          classify,
                                          axis=1,
                                          args=(train_tree,'0') ) # <---- train_data tree

print 'Accuracy is ' + str( sum(test_data['left']==test_data['predicted2'] ) / (1.0*len(test_data.index)) )
