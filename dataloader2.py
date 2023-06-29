#import libraries 
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from dataset import BalancedBatchSampler
import torch
from sklearn import preprocessing
from skimage import io
from skimage import color
from PIL import Image as im
import pandas as pd

#define DataLoader class
class CustomDataset(Dataset):

    '''
    A custom dataset class for loading image data.

    Args:
        label_paths (pandas.DataFrame): A DataFrame containing the image paths and labels.
        image_shape (tuple): The desired shape of the images.
        transforms (torchvision.transforms.Compose): A list of transforms to apply to the images.
    '''

    def __init__(self, label_paths,image_shape,transforms):

        # Store the image paths and labels
        self.label_paths = label_paths
        self.labels=label_paths.label.values

        # Encode the labels
        label_encoder = preprocessing.LabelEncoder()
        labels_encoded=label_encoder.fit_transform(self.labels)
        self.labels_encoded=torch.tensor(list(labels_encoded))

        # Store the transforms
        self.transform = transforms

    def __getitem__(self, index):

        '''
        Fetches an item from the dataset.

        Args:
            index (int): The index of the item to fetch.

        Returns:
            (torch.Tensor, torch.Tensor): The image and label.
        '''
        # Get the image path and label
        label = torch.tensor(self.labels_encoded[index])
        path=self.label_paths.path.values[index]

        # Read the image
        img=io.imread(path)
        img = im.fromarray(img)

        # Apply transforms
        img=self.transform(img)

        return img, label


    def __len__(self):
        '''
        Returns the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        '''
        return len(self.label_paths.index)
    
class valDataset(Dataset):
    
    '''
    A custom dataset class for loading validation image data.

    Args:
        valdataframe (pandas.DataFrame): A DataFrame containing the image paths and labels.
        path (str): The directory containing the images.
        image_shape (tuple): The desired shape of the images.
        transforms (torchvision.transforms.Compose): A list of transforms to apply to the images.
    '''

    def __init__(self, valdataframe,path,image_shape,transforms):

        # Store the image paths and labels
        self.img1_paths=valdataframe.img1_name.values
        self.img2_paths=valdataframe.img2_name.values
        self.path=path
        self.labels=torch.tensor(list(valdataframe.label.values))

        # Store the transforms
        self.transform = transforms
        
    
    def __getitem__(self, index):

        '''
        Fetches an item from the dataset.

        Args:
            index (int): The index of the item to fetch.

        Returns:
            (torch.Tensor, torch.Tensor): The image and label.
        '''

        # Get the image paths and label
        img1_path=self.img1_paths[index]
        img2_path=self.img2_paths[index]
        label=self.labels[index]

        # Read the images
        img1=io.imread(os.path.join(self.path,img1_path))
        img1 = im.fromarray(img1)
        img2=io.imread(os.path.join(self.path,img2_path))
        img2 = im.fromarray(img2)
        
        # Apply the transforms
        img1=self.transform(img1)
        img2=self.transform(img2)
        
        return (img1,img2),label
    
    def __len__(self):
        '''
        Returns the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        '''
        return len(self.labels)  



if __name__=="__main__":
    label_paths=pd.read_csv("./dataset/val.csv")
    dataset = valDataset(label_paths,"./dataset/val")
    data_loader = DataLoader(dataset, batch_size=32)
    
    for i,x in enumerate(data_loader):
        print(len(x))
