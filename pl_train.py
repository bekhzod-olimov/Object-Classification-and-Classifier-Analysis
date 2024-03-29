# Import libraries
import torch, torchmetrics, wandb, timm, argparse, yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from torchvision import transforms as tfs
from torchvision.datasets import ImageFolder, CIFAR100, CIFAR10

class CustomDataset(pl.LightningDataModule):
    
    """
    
    This class gets several arguments and returns train, validation, and test dataloaders.
    
    Parameters:
    
        root      - path to the directory with images, str;
        bs        - mini batch size, int;
        im_dims   - input image dimensions, tuple -> int.
        
    Outputs:
    
        tr_dl     - train dataloader, torch dataloader object;
        val_dl    - validation dataloader, torch dataloader object;
        test_dl   - test dataloader, torch dataloader object;
    
    """
    
    def __init__(self, root, bs, im_dims = (224, 224)):
        
        super().__init__()
        self.root, self.bs = root, bs

        # Initialize transformations
        self.transform = tfs.Compose([
                         # Resize image dimensions
                         tfs.Resize((im_dims)),
                         # Transform grayscale (1 channel) images into 3 channels
                         tfs.Grayscale(num_output_channels = 3),
                         # Transform PIL images into tensor objects
                         tfs.ToTensor()]
                         )
    
    def check(self, path): 
        
        """
        
        This function gets an image path and checks wheter it is a valid image file or not.
        
        Parameter:
        
            path  - an image path, str.
            
        Output:
        
            valid - whether or not the path is a valid image, bool.
        
        """
        
        # Initialize a list with valid image types
        valid_types = [".png", ".jpg", ".jpeg"]
        for valid_type in valid_types:
            if valid_type in path.lower():
                return True
        return False
    
    def data_setup(self):
        
        """
        
        This function reads images from folder, splits the data and returns class names and number of classes in the dataset.
        
        Outputs:
            
            cls_names    - name of classes in the dataset, list;
            num_classes  - number of classes in the dataset, int.
        
        """
        
        self.ds = ImageFolder(root = self.root, transform = self.transform, is_valid_file = self.check)

        # Get class names
        cls_names = list(self.ds.class_to_idx.keys())
        
        # Get number of classes
        num_classes = len(cls_names)
        
        # Get length of the dataset
        ds_len = len(self.ds)
        
        # Get length for train and validation datasets
        tr_len, val_len = int(ds_len * 0.8), int(ds_len * 0.1) 
        
        # Split the dataset into train, validation, and test datasets
        self.tr_ds, self.val_ds, self.test_ds = random_split(self.ds, [tr_len, val_len, ds_len - (tr_len + val_len)])
        
        print(f"Number of train set images: {len(self.tr_ds)}")
        print(f"Number of validation set images: {len(self.val_ds)}")
        print(f"Number of test set images: {len(self.test_ds)}\n")
        
        tr_dl = DataLoader(self.tr_ds, batch_size = self.bs, shuffle = True)
        val_dl = DataLoader(self.val_ds, batch_size = self.bs, shuffle = False)
        test_dl = DataLoader(self.test_ds, batch_size = self.bs, shuffle = False)
        
        return tr_dl, val_dl, test_dl, cls_names, num_classes

class CIFAR10DataModule(pl.LightningDataModule):
        
    """
    
    This class gets several arguments and returns train, validation, and test dataloaders of CIFAR10 dataset.
    
    Parameters:
    
        bs        - batch size, int;
        data_dir  - path to save the downloaded data, str.
        
    Outputs:
    
        tr_dl     - train dataloader, torch dataloader object;
        val_dl    - validation dataloader, torch dataloader object;
        test_dl   - test dataloader, torch dataloader object;
    
    """
    
    def __init__(self, bs, data_dir: str = './'):
        super().__init__()
        
        self.data_dir, self.bs, self.dims, self.num_classes = data_dir, bs, (3, 32, 32), 10
        
        # Initialize transformations
        self.transform = tfs.Compose([
                                      # Resize image dimensions  
                                      tfs.Resize((224, 224)),
                                      # Transform PIL imagens to tensor objects
                                      tfs.ToTensor(),
                                      # Normalize tensor values
                                      tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
        
    def data_setup(self):
        
        """
        
        This function gets data and returns metadata information of the dataset and dataloaders.
        
        Outputs:
        
            tr_dl       - train dataloader, pytorch dataloader object;
            val_dl      - validation dataloader, pytorch dataloader object;
            test_dl     - test dataloader, pytorch dataloader object;\
            cls_names   - class names of the dataset, list;
            num_classes - number of classes in the dataset, int.
        
        """
        
        # Get dataset for training process
        self.ds = CIFAR10(self.data_dir, train = True, download = True, transform = self.transform)
        
        # Get dataset for inference 
        self.test_ds = CIFAR10(self.data_dir, train = False, download = True, transform = self.transform)
        
        # Split the dataset into train and validation datasets
        self.tr_ds, self.val_ds = random_split(self.ds, [int(len(self.ds) * 0.9), int(len(self.ds)) - int(len(self.ds) * 0.9)])
        
        print(f"Number of train set images: {len(self.tr_ds)}")
        print(f"Number of validation set images: {len(self.val_ds)}")
        print(f"Number of test set images: {len(self.test_ds)}\n")

        # Get class names and number of classes
        cls_names, num_classes = None, 10

        # Initialize train, validation, and test dataloaders
        tr_dl = DataLoader(self.tr_ds, batch_size = self.bs, shuffle = True)
        val_dl = DataLoader(self.val_ds, batch_size = self.bs, shuffle = False)
        test_dl = DataLoader(self.test_ds, batch_size = self.bs, shuffle = False)
        
        return tr_dl, val_dl, test_dl, cls_names, num_classes
    
class LitModel(pl.LightningModule):
    
    """"
    
    This class gets several arguments and returns a model for training.
    
    Parameters:
    
        input_shape  - shape of input to the model, tuple -> int;
        model_name   - name of the model from timm library, str;
        num_classes  - number of classes to be outputed from the model, int;
        lr           - learning rate value, float.
    
    """
    
    def __init__(self, input_shape, model_name, num_classes, lr = 2e-4):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters()
        self.lr = lr
        self.accuracy = torchmetrics.Accuracy(task = "multiclass", num_classes = num_classes)
        self.model = timm.create_model(model_name, pretrained = True, num_classes = num_classes)

    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr = self.lr)
    
    def forward(self, x): return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Train metrics
        preds = torch.argmax(logits, dim = 1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step = False, on_epoch = True, logger = True)
        self.log('train_acc', acc, on_step = False, on_epoch = True, logger = True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Validation metrics
        preds = torch.argmax(logits, dim = 1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar = True)
        self.log('val_acc', acc, prog_bar = True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Test metrics
        preds = torch.argmax(logits, dim = 1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, prog_bar = True)
        self.log('test_acc', acc, prog_bar = True)
        
        return loss

class ImagePredictionLogger(Callback):
    
    def __init__(self, val_samples, cls_names = None, num_samples = 4):
        super().__init__()
        self.num_samples, self.cls_names = num_samples, cls_names
        self.val_imgs, self.val_labels = val_samples
        
    def on_validation_epoch_end(self, trainer, pl_module):
        
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device = pl_module.device)
        val_labels = self.val_labels.to(device = pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        
        # Log the images as wandb Image
        if self.cls_names != None:
            trainer.logger.experiment.log({
                "Sample Validation Prediction Results":[wandb.Image(x, caption = f"Predicted class: {self.cls_names[pred]}, Ground truth class: {self.cls_names[y]}") 
                               for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                     preds[:self.num_samples], 
                                                     val_labels[:self.num_samples])]})

        
def run(args):
    
    """
    
    This function runs the main script based on the arguments.
    
    Parameter:
    
        args - parsed arguments.
        
    Output:
    
        train process.
    
    """
    
    # Get train arguments    
    argstr = yaml.dump(args.__dict__, default_flow_style = False)
    print(f"\nTraining Arguments:\n\n{argstr}")
    
    if args.dataset_name == "custom":
        dm = CustomDataset(args.root, bs = args.batch_size, im_dims = args.inp_im_size)
        tr_dl, val_dl, test_dl, cls_names, num_classes = dm.data_setup()
    elif args.dataset_name == "cifar10":
        dm = CIFAR10DataModule(64)
        tr_dl, val_dl, test_dl, cls_names, num_classes = dm.data_setup()

    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(val_dl))
    val_imgs, val_labels = val_samples[0], val_samples[1]

    # model = LitModel(args.inp_im_size, args.model_name, num_classes) if args.dataset_name == 'custom' else LitModel((32, 32), args.model_name, num_classes)
    model = LitModel(args.inp_im_size, args.model_name, num_classes) 

    # Initialize wandb logger
    wandb_logger = WandbLogger(project = "classification", job_type = "train", name = f"{args.model_name}_{args.dataset_name}_{args.batch_size}_{args.learning_rate}")

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs = args.epochs, gpus = args.devices, accelerator="gpu", devices = args.devices, strategy = "ddp", logger = wandb_logger,
                         callbacks = [EarlyStopping(monitor = "val_loss", mode = "min"), ImagePredictionLogger(val_samples, cls_names),
                                      ModelCheckpoint(monitor = "val_loss", dirpath = args.save_model_path, filename = f"{args.model_name}_best")])

    # Train the model
    trainer.fit(model, tr_dl, val_dl)
    # Test the model
    trainer.test(dataloaders = test_dl)
    # Close wandb run
    wandb.finish()
    
if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = 'Image Classification Training Arguments')
    
    # Add arguments to the parser
    parser.add_argument("-r", "--root", type = str, default = 'path/to/your/data', help = "Path to the data")
    parser.add_argument("-bs", "--batch_size", type = int, default = 64, help = "Mini-batch size")
    parser.add_argument("-is", "--inp_im_size", type = tuple, default = (224, 224), help = "Input image size")
    parser.add_argument("-dn", "--dataset_name", type = str, default = 'custom', help = "Dataset name for training")
    parser.add_argument("-mn", "--model_name", type = str, default = 'rexnet_150', help = "Model name for backbone")
    # parser.add_argument("-mn", "--model_name", type = str, default = 'vit_base_patch16_224', help = "Model name for backbone")
    # parser.add_argument("-mn", "--model_name", type = str, default = 'vgg16_bn', help = "Model name for backbone")
    parser.add_argument("-d", "--devices", type = int, default = 3, help = "Number of GPUs for training")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 3e-5, help = "Learning rate value")
    parser.add_argument("-e", "--epochs", type = int, default = 20, help = "Train epochs number")
    parser.add_argument("-sm", "--save_model_path", type = str, default = 'saved_models', help = "Path to the directory to save a trained model")
    parser.add_argument("-sr", "--save_results_path", type = str, default = 'results', help = "Path to the directory to save the train results")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)
