# Import library
from torchvision import transforms as tfs

def get_transforms(train = False):
    
    '''
    
    This function gets argment to distinguish train or validation transformations and return transforms.
    
    Argument:
        
        train - train transformation if True, else validation transformations.
        
    Outputs:
    
        t_tr  - train transformations;
        t_val - validation transformations;
        
    '''
    
    # Train transformations
    t_tr = tfs.Compose([tfs.Resize((224,224)),
                       tfs.RandomCrop((120, 120)),
                       tfs.RandomHorizontalFlip(p=0.3),
                       tfs.RandomRotation(degrees=15),
                       tfs.RandomVerticalFlip(p=0.3),
                       tfs.Grayscale(num_output_channels=3),
                       tfs.ToTensor()])
    
    # Validation transformations
    t_val = tfs.Compose([tfs.Resize((224,224)),
                         tfs.Grayscale(num_output_channels=3),
                         tfs.ToTensor()])
    
    if train: return t_tr
    else: return t_val
