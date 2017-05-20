#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
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

print(score)
print(acc)
### it's all yours from here forward!  


