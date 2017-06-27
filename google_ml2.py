# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:39:18 2017

@author: B6519563
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
"""
print (iris.feature_names)
print (iris.target_names)
print (iris.data[0])
print (iris.target[0])
for i in range(len(iris.target)):
    print ('Example %d: label %s, features %s' % (i, iris.target[i], iris.data[i])) 
"""
test_idx = [0,50,100]

#training data
train_target = np.delete(iris.target, test_idx) #remove test from data
train_data = np.delete(iris.data, test_idx)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print (test_target)
print (clf.predict(test_data))