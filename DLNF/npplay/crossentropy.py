import numpy as np

# interesting to compare my code for cross-entropy calc 
# vs real vectorzed code:

# THis is for binary output Yes No -- so Y vector is just 0s and 1s

def cross_entropyMineNonnVectorized(Y, P):
    result = 0
    for i in range(len(Y)):
        if (Y[i] == 1):
            result -= np.log(P[i])
        else:
            result -= np.log(1-P[i])
    
    return result
def cross_entropyGood(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))