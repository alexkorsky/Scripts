#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
data_dict.pop("TOTAL", 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

from operator import itemgetter
sortedData = sorted(data, key=itemgetter(1))


### your code below
for point in data:
    salary = point[0]


    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


