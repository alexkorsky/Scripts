import numpy as np

# this line reads binary.csv
from data_prep import features, targets, features_test, targets_test

# this is a network with 3-point input to hidden layer with
# two nodes and then going to signle output node
np.random.seed(21)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

    
# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape
last_loss = None
# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # TODO: Calculate the output
        hidden_layer_input = np.dot(x, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
        output = sigmoid(output_layer_in)
        
        ## Backward pass ##
        # TODO: Calculate the network's prediction error
        error = y - output

        output_error_term = error * output * (1 - output)

        # TODO: Calculate error term for hidden layer
        # hidden_term = SUM_k(W_jk * error_k * sigmoid_prime(h_j))
        # where K is number of output nodes, J is number of hidden nodes, 
        # h_j is function calcualted on hidden node j before it is passed to sigmoid

        # So  hidden_term is an array of J elements
        # W_jk is weights_hidden_output
        # sigmoid_prime(h_j) = hidden_layer_output * (1 - hidden_layer_output)
        # here we only have one output so K = 1, J = 2

        # TODO: Calculate error term for hidden layer
        hidden_error_term = np.dot(output_error_term, weights_hidden_output) * \
                    hidden_layer_output * (1 - hidden_layer_output)

        
        # TODO: Update the change in weights
        del_w_hidden_output += output_error_term * hidden_layer_output

        # TODO: Calculate change in weights for input layer to hidden layer
        del_w_input_hidden += hidden_error_term * x[:, None]


    # TODO: Update weights
    weights_input_hidden += learnrate * del_w_input_hidden / n_records
    weights_hidden_output += learnrate * del_w_hidden_output / n_records
    
    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))