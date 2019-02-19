import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer
import sys


def objective(params):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""

    # Keep track of evals
    global ITERATION

    ITERATION += 1

    # Make sure parameters that need to be integers are integers
    for parameter_name in ["hidden_nodes"]:
        params[parameter_name] = int(params[parameter_name])

    start = timer()
    #########
    N_i = train_features.shape[1]
    network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)
    losses = {"train": [], "validation": []}

    for ii in range(iterations):
        # Go through a random batch of 128 records from the training data set
        batch = np.random.choice(train_features.index, size=128)
        X, y = train_features.ix[batch].values, train_targets.ix[batch]["cnt"]

        network.train(X, y)

    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T, train_targets["cnt"].values)
    val_loss = MSE(network.run(val_features).T, val_targets["cnt"].values)
    #############

    run_time = timer() - start

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, "a")
    writer = csv.writer(of_connection)
    writer.writerow([val_loss, params, ITERATION, n_estimators, run_time])

    # Dictionary with information for evaluation
    return {
        "loss": val_loss,
        "params": params,
        "iteration": ITERATION,
        "train_time": run_time,
        "status": STATUS_OK,
    }
