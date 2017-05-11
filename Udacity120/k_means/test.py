# -*- coding: utf-8 -*-
import numpy
    
max = 0
min = 0
def scale(x):
    global max
    global min
    if (max == min):
        return x
    else:
        return (x - min) / (max - min)
        
def featureScaling(arr):

    global max
    global min
    max = numpy.amax(arr, axis=0)   # Maxima along the first axis
    min = numpy.amin(arr, axis=0)
    
    return list(map(scale, arr))

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print (featureScaling(data))
