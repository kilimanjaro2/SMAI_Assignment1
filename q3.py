import csv
import collections
import sys
import numpy as np

class DecisionTree:
	"""Binary tree implementation with true and false branch. """
	def __init__(self, col=-1, value=None, trueBranch=None, falseBranch=None, results=None):
		self.col = col
		self.value = value
		self.trueBranch = trueBranch
		self.falseBranch = falseBranch
		self.results = results # None for nodes, not None for leaves

def divideSet(rows, column, value):
	splittingFunction = None
	if isinstance(value, int) or isinstance(value, float): # for int and float values
		splittingFunction = lambda row : row[column] >= value
	else: # for strings
		splittingFunction = lambda row : row[column] == value
	list1 = [row for row in rows if splittingFunction(row)]
	list2 = [row for row in rows if not splittingFunction(row)]
	return (list1, list2)

def uniqueCounts(rows):
	results = {}
	for row in rows:
		r = row[-1]
		if r not in results: results[r] = 0
		results[r] += 1
	return results

def entropy(rows):
	from math import log
	log2 = lambda x: log(x)/log(2)
	results = uniqueCounts(rows)
	entr = 0.0
	for r in results:
		p = float(results[r])/len(rows)
		entr -= p*log2(p)
	return entr

def variance(rows):
	if len(rows) == 0: return 0
	data = [float(row[len(row) - 1]) for row in rows]
	mean = sum(data) / len(data)
	variance = sum([(d-mean)**2 for d in data]) / len(data)
	return variance

def growDecisionTreeFrom(rows, evaluationFunction=entropy):
	"""Grows and then returns a binary decision tree.
	evaluationFunction: entropy or gini"""
	if len(rows) == 0: return DecisionTree()
	currentScore = evaluationFunction(rows)
	bestGain = 0.0
	bestAttribute = None
	bestSets = None
	columnCount = len(rows[0]) - 1  # last column is the result/target column
	for col in range(0, columnCount):
		columnValues = [row[col] for row in rows]

		for value in columnValues:
			if value != 'nan':
				(set1, set2) = divideSet(rows, col, value)
				p = float(len(set1)) / len(rows)
				gain = currentScore - p*evaluationFunction(set1) - (1-p)*evaluationFunction(set2)
				#print gain
				#print bestGain
				if gain>bestGain and len(set1)>0 and len(set2)>0:
					bestGain = gain
					bestAttribute = (col, value)
					bestSets = (set1, set2)

	if bestGain > 0:
		trueBranch = growDecisionTreeFrom(bestSets[0])
		falseBranch = growDecisionTreeFrom(bestSets[1])
		return DecisionTree(col=bestAttribute[0], value=bestAttribute[1], trueBranch=trueBranch, falseBranch=falseBranch)
	else:
		return DecisionTree(results=uniqueCounts(rows))

def prune(tree, minGain, evaluationFunction=entropy, notify=False):
	"""Prunes the obtained tree according to the minimal gain (entropy or Gini). """
	# recursive call for each branch
	if tree.trueBranch.results == None: prune(tree.trueBranch, minGain, evaluationFunction, notify)
	if tree.falseBranch.results == None: prune(tree.falseBranch, minGain, evaluationFunction, notify)

	# merge leaves (potentionally)
	if tree.trueBranch.results != None and tree.falseBranch.results != None:
		tb, fb = [], []

		for v, c in tree.trueBranch.results.items(): tb += [[v]] * c
		for v, c in tree.falseBranch.results.items(): fb += [[v]] * c

		p = float(len(tb)) / len(tb + fb)
		delta = evaluationFunction(tb+fb) - p*evaluationFunction(tb) - (1-p)*evaluationFunction(fb)
		if delta < minGain:
			if notify: print('A branch was pruned: gain = %f' % delta)
			tree.trueBranch, tree.falseBranch = None, None
			tree.results = uniqueCounts(tb + fb)

def classify(observations, tree, dataMissing=False):
	"""Classifies the observationss according to the tree.
	dataMissing: true or false if data are missing or not. """

	def classifyWithoutMissingData(observations, tree):
		if tree.results != None:  # leaf
			return tree.results
		else:
			v = observations[tree.col]
			branch = None
			if isinstance(v, int) or isinstance(v, float):
				if v >= tree.value: branch = tree.trueBranch
				else: branch = tree.falseBranch
			else:
				if v == tree.value: branch = tree.trueBranch
				else: branch = tree.falseBranch
		#return classifyWithoutMissingData(observations, branch)
		return classifyWithoutMissingData(observations, branch)

	return classifyWithoutMissingData(observations, tree)

def plot(decisionTree):
	"""Plots the obtained decision tree. """
	def toString(decisionTree, indent=''):
		if decisionTree.results != None:  # leaf node
			return str(decisionTree.results)
		else:
			if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
				decision = 'Column %s: x >= %s?' % (decisionTree.col, decisionTree.value)
			else:
				decision = 'Column %s: x == %s?' % (decisionTree.col, decisionTree.value)
			trueBranch = indent + 'yes -> ' + toString(decisionTree.trueBranch, indent + '\t')
			falseBranch = indent + 'no  -> ' + toString(decisionTree.falseBranch, indent + '\t')
			return (decision + '\n' + trueBranch + '\n' + falseBranch)

	print(toString(decisionTree))

if __name__ == '__main__':

	# Select the example you want to classify
	example = 2

	# All examples do the following steps:
	# 	1. Load training data
	# 	2. Let the decision tree grow
	# 	4. Plot the decision tree
	# 	5. classify without missing data
	# 	6. Classifiy with missing data
	# 	(7.) Prune the decision tree according to a minimal gain level
	# 	(8.) Plot the pruned tree

	if example == 1:
		example = 1
	else:
		input_name = str(sys.argv[1])
		unproc_data = np.genfromtxt(input_name,delimiter=",",dtype=None)
		dump = np.copy(unproc_data[:,6])
		unproc_data[:, 6] = unproc_data[:, 9]
		unproc_data[:, 9] = dump
		training_size = unproc_data.shape[0]
		i = 0
		while i < training_size:
			if unproc_data[i][6] == "low" :
				unproc_data[i][6] = float(0.0)
			elif unproc_data[i][6] == "medium" :
				unproc_data[i][6] = float(1.0)
			elif unproc_data[i][6] == "high" :
				unproc_data[i][6] = float(2.0)
			i += 1
		total_size = training_size
		training_size *= 3
		training_size /= 5
		training_size = 1000 # ***************************************** #
		#total_size = 30
		trainingData = unproc_data[:training_size,:]
		testData = unproc_data[training_size:, :9]
		testFlag = unproc_data[training_size:, 9]
		decisionTree = growDecisionTreeFrom(trainingData)

		prune(decisionTree, 0.5, notify=True) # notify, when a branch is pruned (one time in this example)
		plot(decisionTree)
		i = training_size
		while i < total_size:
			print testData[i-training_size]
			print testFlag[i-training_size]
			print(classify(testData[i - training_size], decisionTree)) # dataMissing=False is the default setting
			i += 1
		#print(classify([None, None, None, 1.5], decisionTree, dataMissing=True)) # no longer unique
