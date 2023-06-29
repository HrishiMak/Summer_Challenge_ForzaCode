#Import necessary libraries
import wandb
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from network import SiameseNet
from sklearn.metrics import roc_auc_score

#Function to

def fit(train_loader,  model, train_loss_fn,
        test_loss_fn, optimizer, scheduler,
        n_epochs, device, log_interval,path,
        metrics=[], start_epoch=0, val_loader=None):

    '''
    Args:
        train_loader (torch.utils.data.DataLoader): The training data loader.
        model (nn.Module): The model to train.
        train_loss_fn (nn.Module): The loss function for training.
        optimizer (torch.optim.Optimizer): The optimizer used to train the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        n_epochs (int): The number of epochs to train for.
        device (str): The device to train on.
        log_interval (int): The number of batches after which to print the training loss.
        path (str): The path to save the model weights.
        metrics (list[Metric]): A list of metrics to track during training.
        start_epoch (int, optional): The epoch to start training from. Defaults to 0.
        val_loader (torch.utils.data.DataLoader, optional): The validation data loader. Defaults to None.
    '''
    
    for epoch in range(0, start_epoch):
        #Update learning rate
        scheduler.step()

    max_auc=0

    # Iterate over epochs
    for epoch in range(start_epoch, n_epochs):
        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, train_loss_fn, optimizer, device, log_interval, metrics)
        epoch_metrics = {'Epoch': epoch, 'Train/Epoch_Train_Loss': train_loss}

        # Print training loss
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        # Evaluate model on validation set
        if val_loader is not None:
            val_model=SiameseNet(model)
            val_loss, metrics, auc = test_epoch(val_loader, val_model, test_loss_fn, device, metrics)
            val_loss /= (len(val_loader)*64)
            epoch_metrics['Validation/Epoch_Val_Loss'] = val_loss
            epoch_metrics['Validation/Epoch_Val_AUC'] = auc

            message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                    val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        print(message)

        # Save model if validation AUC is improved
        if auc>max_auc:
            max_auc=auc
            torch.save(model.state_dict(), path)

        scheduler.step()
        wandb.log(epoch_metrics)
                    
    wandb.finish()


def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics):

    """
    Trains the model for one epoch.

    Args:
        train_loader (torch.utils.data.DataLoader): The training data loader.
        model (nn.Module): The model to train.
        loss_fn (nn.Module): The loss function for training.
        optimizer (torch.optim.Optimizer): The optimizer used to train the model.
        device (str): The device to train on.
        log_interval (int): The number of batches after which to print the training loss.
        metrics (list[Metric]): A list of metrics to track during training.
    """

    # Initialize metrics
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0
    
    train_loop = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(train_loop):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        data = tuple(d.to(device) for d in data)
        if target is not None:
            target = target.to(device)

        #Zero the gradients
        optimizer.zero_grad()

        #Forward pass
        outputs = model(*data)

        #Compute loss
        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

        # Log the loss
        train_loop.set_description('Train Loss:{:.4f}'.format(loss.item()))
        wandb.log({"Train/Continuous_Train_Loss": loss.item()})
        losses.append(loss.item())
        total_loss += loss.item()

        #Backpropogate
        loss.backward()

        #Update optimizer
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        #Print training loss
        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

           # print(message)
            losses = []

    # Return the average loss and metrics
    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, device, metrics):

    """
    Evaluates the model on the validation set.

    Args:
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        model (nn.Module): The model to evaluate.
        loss_fn (nn.Module): The loss function for evaluation.
        device (str): The device to evaluate on.
        metrics (list[Metric]): A list of metrics to track during evaluation.

    Returns:
        The average loss and metrics.
    """

    with torch.no_grad():
        for metric in metrics:
            metric.reset()

        # Set model to evaluation mode
        model.eval()

        #Initialize losses
        val_loss = 0
        embeddings_1=[]
        embeddings_2=[]
        labels=[]
        
        val_loop = tqdm(val_loader)
        #Iterate over batches
        for batch_idx, (data, target) in enumerate(val_loop):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            
            #Send data to device
            data = tuple(d.to(device) for d in data)
            if target is not None:
                target = target.to(device)

            #Forward pass
            outputs = model(data[0],data[1])
            embeddings_1.extend(outputs[0].cpu().numpy())
            embeddings_2.extend(outputs[1].cpu().numpy())
            distances = np.power((np.array(embeddings_2) - np.array(embeddings_1)),2).sum(1)
            labels.extend(target.cpu().numpy())
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()
            val_loop.set_description('Val Loss:{:.4f}'.format(loss.item()))
            
            #Update metrics
            for metric in metrics:
                metric(outputs, target, loss_outputs)
        auc = roc_auc_score(np.array(labels),1-nn.Sigmoid()(torch.from_numpy(distances)))
    
    #Return the average loss and metrics
    return val_loss, metrics, auc
