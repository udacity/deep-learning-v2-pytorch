import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim


def _get_loss_acc(model, train_loader, valid_loader):
    """
    Get losses and validation accuracy of example neural network
    """
    n_epochs = 2
    learning_rate = 0.001
    
    # Training loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # Measurements used for graphing loss
    loss_batch = []

    for epoch in range(1, n_epochs+1):
        # initialize var to monitor training loss
        train_loss = 0.0
        ###################
        # train the model #
        ###################
        for data, target in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record average batch loss 
            loss_batch.append(loss.item())
             
    # after training for 2 epochs, check validation accuracy 
    correct = 0
    total = 0
    for data, target in valid_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # get the predicted class from the maximum class score
        _, predicted = torch.max(output.data, 1)
        # count up total number of correct labels
        # for which the predicted and true labels are equal
        total += target.size(0)
        correct += (predicted == target).sum()
      
    # calculate the accuracy
    # to convert `correct` from a Tensor into a scalar, use .item()
    valid_acc = correct.item() / total

    # return model stats
    return loss_batch, valid_acc


def compare_init_weights(
        model_list,
        plot_title,
        train_loader,
        valid_loader,
        plot_n_batches=100):
    """
    Plot loss and print stats of weights using an example neural network
    """
    colors = ['r', 'b', 'g', 'c', 'y', 'k']
    label_accs = []
    label_loss = []

    assert len(model_list) <= len(colors), 'Too many initial weights to plot'

    for i, (model, label) in enumerate(model_list):
        loss, val_acc = _get_loss_acc(model, train_loader, valid_loader)

        plt.plot(loss[:plot_n_batches], colors[i], label=label)
        label_accs.append((label, val_acc))
        label_loss.append((label, loss[-1]))

    plt.title(plot_title)
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    print('After 2 Epochs:')
    print('Validation Accuracy')
    for label, val_acc in label_accs:
        print('  {:7.3f}% -- {}'.format(val_acc*100, label))
    print('Training Loss')
    for label, loss in label_loss:
        print('  {:7.3f}  -- {}'.format(loss, label))
        

def hist_dist(title, distribution_tensor, hist_range=(-4, 4)):
    """
    Display histogram of values in a given distribution tensor
    """
    plt.title(title)
    plt.hist(distribution_tensor, np.linspace(*hist_range, num=len(distribution_tensor)/2))
    plt.show()