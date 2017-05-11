#!/usr/bin/python

import numpy

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    # cleaned_data = outlierCleaner( predictions, ages_train, net_worths_train )

    #sortedErrors = numpy.array(predictions - net_worths)
    #numpy.sort(sortedErrors)
    pred = predictions[:,0]
    ag = ages[:,0]
    nw = net_worths[:,0]
    
    errors = (pred-nw) ** 2

    tuples = list(zip(pred, ag, nw, errors))

    
    from operator import itemgetter
    tuples2 = sorted(tuples, key=itemgetter(3))
    
    cleaned_data = tuples2[0:81]

    ### your code goes here

    
    return cleaned_data

