import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import BalancedBatchSampler
from datasets import CustomDataset, valDataset
from trainer import fit
from network import ViT
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
        #transforms.RandomPerspective(distortion_scale=0.1, p=0.5, interpolation=transforms.InterpolationMode.BILINEAR),
        #transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize((0.8686,),(0.1675,))
    ])
    label_paths = pd.read_csv("/kaggle/working/Handwriting_verification/label_path.csv")
    dataset = CustomDataset(label_paths,(28,280),transform)
    labels = label_paths.label.values
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_encoded = torch.tensor(list(labels_encoded))
    train_batch_sampler = BalancedBatchSampler(labels_encoded, n_classes=8, n_samples=4)
    label_paths = pd.read_csv("/kaggle/input/am-dataset/dataset/val.csv")
    test_dataset = valDataset(label_paths,"/kaggle/input/am-dataset/dataset/val",(28,280),transform)
    kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available else {}
    # creating dataloaders for train and val
    online_train_loader = DataLoader(dataset, batch_sampler=train_batch_sampler, **kwargs) 
    test_loader = DataLoader(test_dataset, batch_size=64, **kwargs)
    print("Dataloaders created!")
    model = ViT(image_size = (28, 280),
        patch_size = configs["patch_size"],
        num_classes = 2,
        dim = configs["dim"],
        depth = configs["depth"],
        heads = configs["heads"],
        mlp_dim = configs["mlp_dim"],
        pool = 'cls',
        channels = 1,
        dim_head = configs["dim_head"]).to(device)
    print("Model Initialized")

    wandb.watch(model)
    # Loss Functions
    assert configs["lossfn"] in ["AllPositivePair"]
    
    if configs["lossfn"] == "AllPositivePair":
        train_loss_fn = OnlineContrastiveLoss(configs["margin"], AllPositivePairSelector())
      
    test_loss_fn = ContrastiveLoss(configs["margin"])
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
        #max_lr = find_best_lr(model, device, online_train_loader, configs["weight_decay"])
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=None, 
                                            epochs=10, steps_per_epoch=len(online_train_loader),
                                            anneal_strategy='cos', cycle_momentum=True, div_factor=25.0)
    else:
        scheduler = None
    print("Optimizer and Scheduler Initialized")
    fit(online_train_loader, model, train_loss_fn, test_loss_fn, optimizer, scheduler, configs["epochs"], device, configs['log_interval'],"/Kaggle/working/Handwriting_verification/best_epoch",val_loader=test_loader)

if __name__ == '__main__':
   main()
