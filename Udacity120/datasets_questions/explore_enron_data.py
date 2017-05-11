#!/usr/bin/python
""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
from feature_format import targetFeatureSplit, featureFormat


print("Alex")
print("Vasya")

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

x  = 0.0
for key in enron_data:
    if (enron_data[key]["total_payments"] == "NaN"):
        x = x + 1
        
print("NaN TotalPayments = " + str(x))        


x  = 0.0
y = 0.0
for key in enron_data:
    if (enron_data[key]["poi"] == 1):
        y= y + 1.0
        if ( enron_data[key]["total_payments"] == "NaN"):
            x = x + 1.0
        
print("Pct = " + str(x / y))

feature_list = ["poi", "salary", "bonus"] 
data_array = featureFormat( enron_data, feature_list )       
label, features = targetFeatureSplit(data_array)

print("Alex2")
print("Vasya2")
