import numpy as np
import math

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    sum = 0
    result = list()
    for x in L:
        sum += math.exp(x)
        
    for x in L:
        result.append(math.exp(x) / sum)
        
    return result
    
print(softmax([1,0]))