from itertools import combinations

import numpy as np
import torch

import torch.nn as nn
import copy
from torch_lr_finder import LRFinder

def pdist(vectors):
    '''
    Calculates the pairwise Euclidean distance between vectors input as tensors.
    '''
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class PairSelector:
    """
    Serves as a base class and defines an interface for getting pairs of embeddings. The get_pairs method needs to be implemented by subclasses.
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()] #pairs where labels of the two elements in each pair are equal.
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()] #pairs where labels of the two elements in each pair are not equal.
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    '''
    Saves a checkpoint of the model state to a file.

    Args:
        state:  model state that you want to save
        filename: name of the file to save the checkpoint. 
    '''
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    '''
    Loads a checkpoint into a model and optimizer.

    Args:
        checkpoint: checkpoint that you want to load
        model: model into which the checkpoint will be saved.
        optimizer: optimizer into which the checkpoint will be saved.
    '''
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def find_best_lr(net, DEVICE, dataloader, weight_decay):
    '''
    Performs a learning rate (LR) range test to find the optimal LR for training a neural network.

    Args:
        net: neural network model
        DEVICE: device (e.g., CPU or GPU) on which the model will be trained
        dataloader: data loader for accessing the training data
        weight_decay: weight decay parameter for regularization
    '''
    exp_net = copy.deepcopy(net).to(DEVICE)
    optimizer = torch.optim.AdamW(exp_net.parameters(), lr=0.00001, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    lr_finder = LRFinder(exp_net, optimizer, criterion, device=DEVICE) #creates an instance of LRFinder with the copied network, optimizer, criterion, and DEVICE.
    lr_finder.range_test(dataloader, end_lr=10, num_iter=200)
    lr_finder.plot()
    min_loss = min(lr_finder.history['loss']) #minimum loss and the LR corresponding to it are determined.
    ler_rate_1 = lr_finder.history['lr'][np.argmin(lr_finder.history['loss'], axis=0)]
    print("Max LR is {}".format(ler_rate_1))

    #same process repeated with updated optimizer
    exp_net = copy.deepcopy(net).to(DEVICE)
    optimizer = torch.optim.Adam(exp_net.parameters(), lr=ler_rate_1/10, weight_decay=weight_decay) #optimizer updated with new lr value
    criterion = nn.CrossEntropyLoss()
    lr_finder = LRFinder(exp_net, optimizer, criterion, device=DEVICE) 
    lr_finder.range_test(dataloader, end_lr=ler_rate_1*10, num_iter=200)
    lr_finder.plot()
    min_loss = min(lr_finder.history['loss'])
    ler_rate_2 = lr_finder.history['lr'][np.argmin(lr_finder.history['loss'], axis=0)]
    print("Max LR is {}".format(ler_rate_2))


    ler_rate = ler_rate_2
    print("Determined Max LR is:", ler_rate)
    
    return ler_rate
