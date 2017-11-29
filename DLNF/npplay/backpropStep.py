import numpy as np

# this is a network with 3-point input to hidden layer with
# two nodes and then going to signle output node

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))
    
x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)

## Backwards pass
## TODO: Calculate output error
error = target - output

# TODO: Calculate error term for output layer
output_error_term = error * sigmoid_prime(output_layer_in)

# TODO: Calculate error term for hidden layer
# hidden_term = SUM_k(W_jk * error_k * sigmoid_prime(h_j))
# where K is number of output nodes, J is number of hidden nodes, 
# h_j is function calcualted on hidden node j before it is passed to sigmoid

# So  hidden_term is an array of J elements
# W_jk is weights_hidden_output
# sigmoid_prime(h_j) = hidden_layer_output * (1 - hidden_layer_output)
# here we only have one output so K = 1, J = 2
'''
hidden_error_term = weights_hidden_output[0] * error * hidden_layer_input[0] + \
weights_hidden_output[1] * error * hidden_layer_input[1]
'''

# TODO: Calculate error term for hidden layer
hidden_error_term = output_error_term * weights_hidden_output * \
                    hidden_layer_output * (1 - hidden_layer_output)

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * output_error_term * hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = learnrate * hidden_error_term * x[:, None]

print(hidden_error_term)
print (x)
print(x[:, None])

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)
