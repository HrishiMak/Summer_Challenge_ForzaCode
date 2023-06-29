import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    '''
    Contrastive loss function.

    Args:
        margin (float): The margin between the positive and negative distances.
    '''

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):

        '''
        Calculates the loss for a batch of embeddings.

        Args:
            output1 (torch.Tensor): The embeddings of the first sample.
            output2 (torch.Tensor): The embeddings of the second sample.
            target (torch.Tensor): The labels of the samples.
            size_average (bool): If True, the loss is averaged over the batch size.

        Returns:
            torch.Tensor: The loss for the batch.
        '''

        # Compute squared distances and losses
        distances = (output2 - output1).pow(2).sum(1)  
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    '''
    Triplet loss function.

    Args:
        margin (float): The margin between the anchor and positive distance, the anchor and negative distance.
    '''

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):

        '''
        Calculates the loss for a batch of embeddings.

        Args:
            anchor (torch.Tensor): The embeddings of the anchor sample.
            positive (torch.Tensor): The embeddings of the positive sample.
            negative (torch.Tensor): The embeddings of the negative sample.
            size_average (bool): If True, the loss is averaged over the batch size.

        Returns:
            torch.Tensor: The loss for the batch.
        '''

        #Calculate distances
        distance_positive = (anchor - positive).pow(2).sum(1)  
        distance_negative = (anchor - negative).pow(2).sum(1)  

        # Calculate loss
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    A contrastive loss function for online training

    Args:
        margin (float): The margin between the positive and negative pairs.
        pair_selector (PairSelector): A PairSelector object that generates the positive and negative pairs.
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):

        '''
        Calculates the loss for a batch of embeddings.

        Args:
            embeddings (torch.Tensor): The batch of embeddings.
            target (torch.Tensor): The batch of labels.

        Returns:
            torch.Tensor: The loss for the batch.
        '''
        
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()

        # Get the distances between the positive and negative pairs
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        
        # Calculate the loss
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    '''
    A triplet loss function for online training.

    Args:
        margin (float): The margin between the anchor and positive distance and the anchor and negative distance.
        triplet_selector (TripletSelector): A TripletSelector object that generates the triplets.
    '''

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        '''
        Calculates the loss for a batch of embeddings.

        Args:
            embeddings (torch.Tensor): The batch of embeddings.
            target (torch.Tensor): The batch of labels.

        Returns:
            torch.Tensor: The loss for the batch.
        '''

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        # Get the distances between the anchor and positive, and anchor and negative
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)

        # Calculate the loss  
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)                   