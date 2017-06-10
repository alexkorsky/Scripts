#!/usr/bin/python


""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
#from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=40)
 
#kernel="linear"

#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data

#features_train = features_train[:(int)(len(features_train)/100)]
#labels_train = labels_train[:(int)(len(labels_train)/100)]

num_features = len(features_train[0])

clf.fit(features_train, labels_train)

print ("Num Features", num_features)

print ("Started Preditction", len(labels_train)-sum(labels_train))

pred=clf.predict(features_test)

print ("End Prediciton", len(labels_train)-sum(labels_train))

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print ("Accuracy ", acc)
print ("Pred 10 ", pred[10])
print ("Pred 26 ", pred[26])
print ("Pred 50 ", pred[50])

#########################################################


