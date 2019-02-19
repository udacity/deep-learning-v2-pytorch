import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(
            0.0, self.input_nodes ** -0.5, (self.input_nodes, self.hidden_nodes)
        )

        self.weights_hidden_to_output = np.random.normal(
            0.0, self.hidden_nodes ** -0.5, (self.hidden_nodes, self.output_nodes)
        )
        self.lr = learning_rate

        # Set self.activation_function to  sigmoid function
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, features, targets):
        """ Train the network on batch of features and targets. 
            Arguments
            ---------
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        """
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):

            #  Do forward pass
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            # Calculate the errors get updated weights that may reduce the errors
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(
                final_outputs,
                hidden_outputs,
                X,
                y,
                delta_weights_i_h,
                delta_weights_h_o,
            )
        # Update the weights
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        """ Implement forward pass here 

            Arguments
            ---------
            X: features batch

        """
        ### Forward pass ###
        # signals into hidden layer
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # signals into final output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        # signals from final output layer
        final_outputs = final_inputs

        return final_outputs, hidden_outputs

    def backpropagation(
        self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o
    ):
        """ Implement backpropagation

            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        """
        ### Backward pass ###

        # Output layer error is the difference between desired target and actual output.
        error = y - final_outputs

        # Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(error, self.weights_hidden_to_output.T)

        # Backpropagated error terms - Replace these values with your calculations.
        output_error_term = error

        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None]
        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        """ Update weights on gradient descent step

            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        """
        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        """ Run a forward pass through the network with input features 

            Arguments
            ---------
            features: 1D array of feature values
        """

        #### Implement the forward pass here ####
        # signals into hidden layer
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # signals into final output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        # signals from final output layer
        final_outputs = final_inputs
        return final_outputs


#########################################################
# Set your hyperparameters here (Second least MSE loss from below table)
##########################################################
iterations = 10600
learning_rate = 0.2887998961705719
hidden_nodes = 10
output_nodes = 1

# Best results for several hyper-parameters
# +--------------------------------------------------------------------------------------------------------------------------------------------------------+
# | loss                            | hyper-parameters                                                                                                         | train_time     |
# +--------------------------------------------------------------------------------------------------------------------------------------------------------+
# | 0.141505344514840   |   {'hidden_nodes': 28, 'iterations': 14800, 'learning_rate': 0.25177603192295606} | 488.2881553 |
# | 0.142963746497879   |   {'hidden_nodes': 10, 'iterations': 10600, 'learning_rate': 0.2887998961705719} | 250.2367154 |
# | 0.144113653142509   |   {'hidden_nodes': 15, 'iterations': 13100, 'learning_rate': 0.2973050618717538} | 456.0543504 |
# | 0.144807735571230   |   {'hidden_nodes': 14, 'iterations': 13900, 'learning_rate': 0.15078816632087946} | 378.9936788 |
# | 0.146030811027299   |   {'hidden_nodes': 20, 'iterations': 9500, 'learning_rate': 0.27244255300385317} | 243.3451003 |
# | 0.149928860322724   |   {'hidden_nodes': 22, 'iterations': 10800, 'learning_rate': 0.2992049147740791} | 329.2540395 |
# | 0.150713738899908   |   {'hidden_nodes': 29, 'iterations': 8600, 'learning_rate': 0.2978565363984502} | 247.3681888 |
# | 0.151329791438927   |   {'hidden_nodes': 15, 'iterations': 10700, 'learning_rate': 0.15798876504923315} | 254.6575523 |
# | 0.155331011477999   |   {'hidden_nodes': 15, 'iterations': 12300, 'learning_rate': 0.15179366560262886} | 292.8279305 |
# | 0.155825899623305   |   {'hidden_nodes': 30, 'iterations': 14800, 'learning_rate': 0.22662035053418858} | 434.0398029 |
# | 0.156869800216103   |   {'hidden_nodes': 26, 'iterations': 7600, 'learning_rate': 0.24963531261600613} | 215.8153883 |
# +-------------------------------------------------------------------------------------------------------------------------------------------------------+
