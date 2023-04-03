# Import libraries
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch, torchmetrics, wandb, timm, argparse, yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from torchvision import transforms as tfs
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

class CustomDataset(pl.LightningDataModule):
    
    """
    
    This class gets several arguments and returns train, validation, and test dataloaders.
    
    Arguments:
    
        root    - path to the directory with images, str;
        bs      - mini batch size, int;
        im_dims - input image dimensions, tuple -> int.
    
    """
    
    def __init__(self, root, bs, im_dims = (224, 224)):
        
        super().__init__()
        self.root, self.bs = root, bs

        # Initialize transformations
        self.transform = tfs.Compose([tfs.Resize((im_dims)),
                         tfs.Grayscale(num_output_channels = 3),
                         tfs.ToTensor()])
    
    def check(self, path): 
        
        valid_types = [".png", ".jpg", ".jpeg"]
        for valid_type in valid_types:
            if valid_type in path.lower():
                return True
        return False
    
    def _setup(self):
        
        self.ds = ImageFolder(root = self.root, transform = self.transform, is_valid_file=self.check)

        # Get class names
        cls_names = list(self.ds.class_to_idx.keys())
        num_classes = len(cls_names)
        
        # Get length of the dataset
        ds_len = len(self.ds)
        tr_len, val_len = int(ds_len * 0.8), int(ds_len * 0.1) 
        # Split the dataset into train and validation datasets
        self.tr_ds, self.val_ds, self.test_ds = torch.utils.data.random_split(self.ds, [tr_len, val_len, ds_len - (tr_len + val_len)])
        
        print(f"Number of train set images: {len(self.tr_ds)}")
        print(f"Number of validation set images: {len(self.val_ds)}")
        print(f"Number of test set images: {len(self.test_ds)}\n")
        
        return cls_names, num_classes

    def train_dataloader(self): return DataLoader(self.tr_ds, batch_size=self.bs, shuffle=True)

    def val_dataloader(self): return DataLoader(self.val_ds, batch_size=self.bs, shuffle=False)

    def test_dataloader(self): return DataLoader(self.test_ds, batch_size=self.bs, shuffle=False)

class LitModel(pl.LightningModule):
    
    def __init__(self, input_shape, model_name, num_classes, learning_rate=2e-4):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
        
    # will be used during inference
    def forward(self, x): return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

class ImagePredictionLogger(Callback):
    
    def __init__(self, val_samples, class_names, num_samples=4):
        super().__init__()
        self.num_samples, self.class_names = num_samples, class_names
        self.val_imgs, self.val_labels = val_samples
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "Sample Validation Prediction Results":[wandb.Image(x, caption=f"Predicted class: {self.class_names[pred]}, Ground truth class: {self.class_names[y]}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
            })

        
def run(args):
    
    """
    
    This function runs the main script based on the arguments.
    
    Argument:
    
        args - parsed arguments.
        
    Output:
    
        train process.
    
    """
    
    # Get train arguments    
    argstr = yaml.dump(args.__dict__, default_flow_style = False)
    print(f"\nTraining Arguments:\n\n{argstr}")
    
    dm = CustomDataset(args.root, bs=args.batch_size, im_dims = args.inp_im_size)
    cls_names, num_classes = dm._setup()

    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(dm.val_dataloader()))
    val_imgs, val_labels = val_samples[0], val_samples[1]

    model = LitModel(args.inp_im_size, args.model_name, num_classes)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project='classification', job_type='train', name=f"{args.model_name}_{args.batch_size}_{args.learning_rate}")
    
    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.devices, logger=wandb_logger,
                         callbacks = [EarlyStopping(monitor='val_loss', mode='min'), ImagePredictionLogger(val_samples, cls_names),
                                      ModelCheckpoint(monitor='val_loss', dirpath=args.save_model_path, filename=f'{args.model_name}_best')])

    trainer.fit(model, dm)

    trainer.test(dataloaders=dm.test_dataloader())

    # Close wandb run
    wandb.finish()
        
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'Object Classification Training Arguments')
    
    # parser.add_argument("-r", "--root", type = str, default = '/home/ubuntu/workspace/bekhzod/triplet-loss-pytorch/pytorch_lightning/data/simple_classification', help = "Path to the data")
    parser.add_argument("-r", "--root", type = str, default = '/home/ubuntu/workspace/bekhzod/class/01.fer/fer2013plus/train', help = "Path to the data")
    parser.add_argument("-bs", "--batch_size", type = int, default = 64, help = "Mini-batch size")
    parser.add_argument("-is", "--inp_im_size", type = tuple, default = (224, 224), help = "Input image size")
    parser.add_argument("-mn", "--model_name", type = str, default = 'rexnet_150', help = "Model name for backbone")
    parser.add_argument("-d", "--devices", type = int, default = 3, help = "Number of GPUs for training")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 3e-5, help = "Learning rate value")
    parser.add_argument("-e", "--epochs", type = int, default = 50, help = "Train epochs number")
    parser.add_argument("-sm", "--save_model_path", type = str, default = 'saved_models', help = "Path to the directory to save a trained model")
    parser.add_argument("-sr", "--save_results_path", type = str, default = 'results', help = "Path to the directory to save the train results")
    
    args = parser.parse_args() 
    
    run(args) 
