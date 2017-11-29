import numpy as np

# Activation (sigmoid) function
def sigmoid(x):
    return (1/(1+np.exp(-1 * x)))

# Output (prediction) formula
def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)

# Error (log-loss) formula
def error_formula(y, output):
    y = np.float_(y)
    output = np.float_(output)
    return -np.sum(y * np.log(output) + (1 - y) * np.log(1 - output))

# Gradient descent step\
# x is a point [xi, .., xn], y i single label 1 or 0, weights is an array, bias is a number
def update_weights(x, y, weights, bias, learnrate):
    result = np.zeros(weights.shape, float)
    for i in range(len(weights)):
        result[i] = weights[i] - learnrate * (output_formula(x, weights, bias) - y) * x[i]
    
    newbias = bias - learnrate * (output_formula(x, weights, bias) - y)
    
    return result, newbias

def update_weights2(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias)
    d_error = -(y - output)
    weights -= learnrate * d_error * x
    bias -= learnrate * d_error
    return weights, bias

def train(features, targets, epochs, learnrate, graph_lines=False):

    print(features)
    print("++++++++++++++++++++")
    print(targets)
        
    errors = []
    n_records, n_features = features.shape
    last_loss = None
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    print("Weights:")
    print(weights)
    bias = 0
    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        

        for x, y in zip(features, targets):
            
            v= np.array(x)
            print("X:")
            print(x)
            print("Y:")
            print(y)
            print("")
           
            output = output_formula(x, weights, bias)
            error = error_formula(y, output)
            # x is a point [xi, .., xn], y i single label 1 or 0, weights is an array, bias is a number
            weights, bias = update_weights(x, y, weights, bias, learnrate)
        
        # Printing out the log-loss error on the training set
        out = output_formula(features, weights, bias)
        loss = np.mean(error_formula(targets, out))
        errors.append(loss)
        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e,"==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            predictions = out > 0.5
            accuracy = np.mean(predictions == targets)
            print("Accuracy: ", accuracy)
        if graph_lines and e % (epochs / 100) == 0:
            display(-weights[0]/weights[1], -bias/weights[1])
            


    # Plotting the solution boundary
    plt.title("Solution boundary")
    display(-weights[0]/weights[1], -bias/weights[1], 'black')

    # Plotting the data
    plot_points(features, targets)
    plt.show()

    # Plotting the error
    plt.title("Error Plot")
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.show()
    

np.random.seed(44)
epochs = 1
learnrate = 0.01

X = np.array([[ 0.78051,   -0.063669 ], \
 [ 0.28774,    0.29139  ], \
 [ 0.40714,    0.17878  ], \
 [ 0.2923,     0.4217   ], \
 [ 0.50922,   0.35256  ], \
 [ 0.27785,    0.10802  ]])
 
Y = np.array([1, 1, 0, 0, 1, 1])

train(X, Y, epochs, learnrate, True)

