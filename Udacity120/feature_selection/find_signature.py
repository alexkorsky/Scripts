#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "rb"))
authors = pickle.load( open(authors_file, "rb") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train_tf = vectorizer.fit_transform(features_train)
features_test_tf  = vectorizer.transform(features_test).toarray()

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train_tf[:150].toarray()
labels_train   = labels_train[:150]

### your code goes here
xxx = features_test[1]

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(features_train, labels_train)
score = clf.score(features_test_tf, labels_test)

#from sklearn.metrics import accuracy_score
#acc = accuracy_score(features_test, labels_test)

sss = clf.feature_importances_

xxx = (vectorizer.get_feature_names())
max = 0.2
myindex = 0
for index in range(len(sss)):
    if (sss[index] > max ):
       # max = sss[index]
       # myindex = index
       print ("Importance: ", sss[index], " Index: ", index, " Word: " , xxx[index])


#print(xxx[myindex])
#print(xxx[34597])

print ("Accuracy: ", score)


