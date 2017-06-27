# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:47:15 2017

@author: B6519563
"""
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.x_train[0])
        best_index = 0
        for i in range(1,len(self.x_train)):
            dist = euc(row, self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]
        
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
#from sklearn.neighbors import KNeighborsClassifier
my_classifier = ScrappyKNN()

my_classifier.fit(x_train, y_train) #leaning

predictions = my_classifier.predict(x_test) #predicting

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, predictions)) #evaluating