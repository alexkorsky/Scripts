#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)

from sklearn import cross_validation
features_Train, features_Test, labels_Train, labels_Test = cross_validation.train_test_split(features, labels,
                                                                                            test_size = 0.3, 
                                                                                            random_state = 42)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_Train, labels_Train)

pred=clf.predict(features_Test)


from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_Test)


score = clf.score(features_Test, labels_Test)



buckets = [0.0] * 29
acc = accuracy_score(buckets, labels_Test)

print(score)
print(acc)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print (classification_report(labels_Test, pred))
print (confusion_matrix(labels_Test, pred))

### your code goes here 


