import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import BCELoss

from utils import BalancedBatchSampler
from datasets import CustomDataset,CustomDataset_emb, valDataset_emb
from trainer_emb import fit
from network import ViT,ClassificationNet,SiameseNet
from losses import OnlineContrastiveLoss
from losses import ContrastiveLoss
from utils import AllPositivePairSelector, find_best_lr
import wandb

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device: ", device)

static_configs = {
    'seed': 42,
    'num_workers': 2,
    'log_interval': 50,
}

hyp_configs = {
    "epochs": 200,
    "batch_size":32,
    "lr": 3e-5,
    "weight_decay": 0.0001,
    "margin": 1.0,
    "dim": 16,
    "patch_size": (14, 14),
    "depth": 12,
    "heads": 4,
    "mlp_dim": 128,
    "dim_head": 16,
    "scheduler_type": "StepLR",
}

wandb.init(project="Hand_writing_verification",entity="am_handwriting", config={**static_configs, **hyp_configs})
configs = wandb.config
print("configs: ", configs)

def main():
    transform = transforms.Compose([
        transforms.Resize((28,280)),
        transforms.ToTensor(),
        transforms.Normalize((0.8686,),(0.1675,))
    ])
    model_emb=torch.load("/kaggle/working/Handwriting_verification/saved_model-0.9266")
    label_paths = pd.read_csv("/kaggle/working/Handwriting_verification/label_path.csv")
    dataset = CustomDataset(label_paths,(28,280),transform)
    kwargs = {'num_workers': configs['num_workers'], 'pin_memory': True} if torch.cuda.is_available else {}
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, **kwargs)
    embeddings_1=[]
    target=[]
    with torch.no_grad():
        model_emb.eval()
        for i,k in enumerate(train_loader):
            data=k[0]
            if not type(k[0]) in (tuple, list):
                data = (k[0],)
            data = tuple(d.to(device) for d in data)
            outputs = model_emb(*data)
            embeddings_1.extend(outputs.cpu().numpy())
            target.extend(k[1].cpu().numpy())
    label_paths = pd.read_csv("/kaggle/input/am-dataset/dataset/val.csv")
    test_dataset = valDataset(label_paths,"/kaggle/input/am-dataset/dataset/val",(28,280),transform)
    test_loader = DataLoader(test_dataset, batch_size=64, **kwargs)
    val_embeddings_1=[]
    val_embeddings_2=[]
    val_target=[]
    with torch.no_grad():
        model_emb.eval()
        for i,k in enumerate(test_loader):
            data=k[0]
            if not type(k[0]) in (tuple, list):
                data = (k[0],)
            data = tuple(d.to(device) for d in data)
            model_siamese=SiameseNet(model_emb)
            outputs = model_siamese(data[0],data[1])
            val_embeddings_1.extend(outputs[0].cpu().numpy())
            val_embeddings_2.extend(outputs[1].cpu().numpy())
            val_target.extend(k[1].cpu().numpy())
    
    label_paths = pd.read_csv("/kaggle/working/Handwriting_verification/label_path.csv")
    dataset = CustomDataset_emb(embeddings_1, target)
    labels_encoded = torch.tensor(list(target))
    train_batch_sampler = BalancedBatchSampler(labels_encoded, n_classes=8, n_samples=4)
    label_paths = pd.read_csv("/kaggle/input/am-dataset/dataset/val.csv")
    test_dataset = valDataset_emb(val_embeddings_1,val_embeddings_2,val_target)
    kwargs = {'num_workers': configs['num_workers'], 'pin_memory': True} if torch.cuda.is_available else {}
    online_train_loader = DataLoader(dataset, batch_sampler=train_batch_sampler, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=64, **kwargs)
    print("Dataloaders created!")
    model = ClassificationNet(40).to(device)
    print("Model Initialized")
    
    # Loss Functions
    train_loss_fn = BCELoss()
    test_loss_fn = BCELoss()
    print("Loss Functions Initialized")
    
    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"])
    
    assert configs["scheduler_type"] in ["StepLR", "ReduceLROnPlateau", "CosineAnnealingLR", "OneCycleLR", None]
    
    if configs["scheduler_type"] == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    elif configs["scheduler_type"] == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.0001,
                                                   threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    elif configs["scheduler_type"] == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=0, last_epoch=-1, verbose=False)
    elif configs["scheduler_type"] == "OneCycleLR":
        max_lr = find_best_lr(model, device, online_train_loader, configs["weight_decay"])
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=None, 
                                            epochs=10, steps_per_epoch=len(online_train_loader),
                                            anneal_strategy='cos', cycle_momentum=True, div_factor=25.0)
    else:
        scheduler = None
    print("Optimizer and Scheduler Initialized")
    fit(online_train_loader, model, train_loss_fn, test_loss_fn, optimizer, scheduler, configs["epochs"], device, configs['log_interval'],"/Kaggle/working/Handwriting_verification/best_epoch",val_loader=test_loader)

if __name__ == '__main__':
   main()
