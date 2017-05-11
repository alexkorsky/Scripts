#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list1 = [poi, feature_1, feature_2]
features_list2 = [poi, feature_1, feature_2, feature_3]
data1 = featureFormat(data_dict, features_list1 )
data2 = featureFormat(data_dict, features_list2 )
poi1, finance_features1 = targetFeatureSplit( data1 )
poi2, finance_features2 = targetFeatureSplit( data2 )

max = numpy.amax(finance_features1, axis=0)   # Maxima along the first axis
min = numpy.amin(finance_features1, axis=0)   # Maxima along the first axis

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(finance_features1)

xxx = scaler.transform(numpy.array([200000., 1000000.]))


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features1:
    plt.scatter( f1, f2 )
plt.show()

for f1, f2, f3 in finance_features2:
    plt.scatter( f1, f2 )
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred

from sklearn.cluster import KMeans


kmeans1 = KMeans(n_clusters=2).fit(finance_features1)
pred1 = kmeans1.predict(finance_features1)


### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
#try:
 #   Draw(pred1, finance_features1, poi1, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
#except NameError:
#    print ("no predictions object named pred found, no clusters to plot")


kmeans2 = KMeans(n_clusters=2).fit(finance_features2)
pred2 = kmeans2.predict(finance_features2)


### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred2, finance_features2, poi2, mark_poi=False, name="clusters2.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print ("no predictions object named pred found, no clusters to plot")