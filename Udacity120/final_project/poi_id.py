#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


researchfeatures = ( 
'bonus',
'deferral_payments',
'deferred_income',
'director_fees',
'email_address',
'exercised_stock_options',
'expenses',
'from_messages',
'from_poi_to_this_person',
'from_this_person_to_poi',
'loan_advances',
'long_term_incentive',
'other',
'poi',
'restricted_stock',
'restricted_stock_deferred',
'salary',
'shared_receipt_with_poi',
'to_messages',
'total_payments',
'total_stock_value'
)

# create map of ids to Person name
id = 0
id_name_dictionary = {}
name_id_dictionary = {}
for key in data_dict:
    if (key != "TOTAL"):
        id_name_dictionary[id] = key
        name_id_dictionary[key] = id
        id = id + 1
    
import matplotlib.pyplot as plt
for feature in researchfeatures:
    print (feature);
    if (feature != 'email_address' and feature == 'bonus'):
        points = []
        colors = []
        labels = []
        for key in data_dict:
            if (key != "TOTAL" and data_dict[key][feature] != 'NaN'):
                points.append(float(data_dict[key][feature]) / 10000)
                labels.append(name_id_dictionary[key])
                if (data_dict[key]['poi'] == 1 ):
                    colors += 'r'
                else:
                    colors += 'b'
            #color += data_dict[key]['poi'] == 1 ? 'r' : 'b'                
      
        plt.xlim(0, int(max(labels)))
        plt.ylim(0, int(max(points)))
        plt.axis([0, int(max(labels)), 0 , int(max(points))]) 

        plt.scatter(labels, points, color=colors )

        plt.show()

    
'''

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

'''