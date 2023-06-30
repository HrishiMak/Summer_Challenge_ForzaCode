import torch
import numpy as np
from network import SiameseNet
from tqdm import tqdm
import wandb
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from utils import AllPositivePairSelector

def fit(train_loader,  model, train_loss_fn,test_loss_fn, optimizer, scheduler, n_epochs, device, log_interval,path, metrics=[],
        start_epoch=0,val_loader=None):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()
    max_auc=0
    for epoch in range(start_epoch, n_epochs):
        

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, train_loss_fn, optimizer, device, log_interval, metrics)
        epoch_metrics = {'Epoch': epoch, 'Train/Epoch_Train_Loss': train_loss}

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        if val_loader is not None:
            val_loss, metrics, auc = test_epoch(val_loader, model, test_loss_fn, device, metrics)
            val_loss /= (len(val_loader)*64)
            epoch_metrics['Validation/Epoch_Val_Loss'] = val_loss
            epoch_metrics['Validation/Epoch_Val_AUC'] = auc

            message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                    val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)
        scheduler.step()
        wandb.log(epoch_metrics)
        if auc>max_auc:
            max_auc=auc
            torch.save(model.state_dict(), path)
            
    wandb.finish()


def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics):
    '''
    Performs the training loop for one epoch, computes the loss, performs backpropagation, updates the model's parameters, and tracks the metrics.
    '''
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0
    
    train_loop = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(train_loop):
        embedding_batch = data
        label_batch = target
        positive,negative = AllPositivePairSelector().get_pairs(embedding_batch, label_batch)
      
        # positive and negative loss values are calculated by computing the squared Euclidean distance between the embeddings of the positive and negative pairs.
        positive_loss = (embedding_batch[positive[:, 0]] - embedding_batch[positive[:, 1]]).pow(2)
        negative_loss =(embedding_batch[negative[:, 0]] - embedding_batch[negative[:, 1]]).pow(2)
      
        data = torch.cat((positive_loss,negative_loss)).type(torch.float32)
        target = torch.cat((torch.ones((positive_loss.shape[0],1)),torch.zeros((negative_loss.shape[0],1))))
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        data = tuple(d.to(device) for d in data)
        if target is not None:
            target = target.to(device)
        
    
        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)
          
        #loss inputs are set to be the outputs of the model, and if the target is not None, it is added to the loss inputs.
        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        train_loop.set_description('Train Loss:{:.4f}'.format(loss.item()))
        wandb.log({"Train/Continuous_Train_Loss": loss.item()})
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward() #backpropagation
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

           # print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, device, metrics):
    '''
    Performs the evaluation loop on the validation data, computes the loss, updates the metrics, and calculates the AUC
    '''
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        values=[]
        labels=[]
        
        val_loop = tqdm(val_loader)
        for batch_idx, (data, target) in enumerate(val_loop):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            data = tuple(d.to(device) for d in data)
            if target is not None:
                target = target.to(device)
            outputs = model(*data)
            target=target.view(-1,1).type(torch.float32)
            #print(target.shape)
            values.extend(outputs.cpu().numpy())
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
            
              
            for metric in metrics:
                metric(outputs, target, loss_outputs)
        auc = roc_auc_score(np.array(labels),np.array(values))
    return val_loss, metrics, auc
