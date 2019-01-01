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
    
def sigmoid(x):
    return (1/(1+np.exp(-1 * x)))
    
# interesting to compare my cross-entropy formula vs. 
print(softmax([1,0]))

x = 2*0.4 + 6*0.6 + (-2)
y = 3*0.4 + 5*0.6 + (-2.2)
z = 5*0.4 + 4*0.6 + (-3)
print(sigmoid(x))
print(sigmoid(y))
print(sigmoid(z))
print(sigmoid(-5))

print(np.full((4,), 45))