# Import libraries
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch

def get_dl(root, bs, t):
    
    '''
    
    This function gets a path to the data and returns class names, number of classes, train dataloader, and validation dataloader.
    
    Arguments:
    
        root         - path to the images;
        bs           - batch size of the dataloaders;
        t            - transformations;
        
    Outputs:
    
        cls_names    - names of the classes in the dataset;
        num_classes  - number of the classes in the dataset;
        tr_dl        - train dataloader;
        val_dl       - validation dataloader.
        
    '''
    
    # Get dataset from the directory
    ds = ImageFolder(root = root, transform = t)
    
    # Get length of the dataset
    ds_length = len(ds)
    
    # Split the dataset into train and validation datasets
    tr_ds, val_ds = torch.utils.data.random_split(ds, [int(ds_length * 0.8), ds_length-int(ds_length * 0.8)])
    print(f"Number of train set images: {len(tr_ds)}")
    print(f"Number of validation set images: {len(val_ds)}\n")
    
    # Get class names
    cls_names = list(ds.class_to_idx.keys())
    # Get total number of classes
    num_classes = len(cls_names)
    
    # Create train and validation dataloaders
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False)
    
    # Return class names, total number of classes, train and validation dataloaders
    return cls_names, num_classes, tr_dl, val_dl
