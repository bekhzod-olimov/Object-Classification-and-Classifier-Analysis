from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch

def get_dl(root, bs, t):
    
    ds = ImageFolder(root=root, transform=t)
    ds_length = len(ds)
    tr_ds, val_ds = torch.utils.data.random_split(ds, [int(ds_length * 0.8), ds_length-int(ds_length * 0.8)])
    print(f"Number of train set images: {len(tr_ds)}")
    print(f"Number of validation set images: {len(val_ds)}")
    
    cls_names = list(ds.class_to_idx.keys())
    num_classes = len(cls_names)
    
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False)
    
    return cls_names, num_classes, tr_dl, val_dl