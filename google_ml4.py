# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:23:54 2017

@author: B6519563
"""

#import a dataset
from sklearn import datasets
iris = datasets.load_iris()

x = iris.data #features
y = iris.target #labels

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)
"""
#tree classifier
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
"""

#KNN
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(x_train, y_train) #leaning

predictions = my_classifier.predict(x_test) #predicting

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, predictions)) #evaluating