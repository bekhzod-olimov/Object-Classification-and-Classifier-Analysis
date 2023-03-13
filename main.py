import torch, argparse, yaml, timm, gdown
from transforms import get_transforms
from dataset import get_dl
from train import train
from tqdm import tqdm

def run(args):
    
    """
    
    This function runs the main script based on the arguments.
    
    Argument:
    
        args - parsed arguments.
    
    """
    
    # Get train arguments    
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print(f"\nTraining Arguments:\n{argstr}")
    
    # Get train and validation transformations 
    train_transformations, valid_transformations= get_transforms(train=True), get_transforms(train=False)
    
    # Get class names, number of classes, train and validation dataloaders
    cls_names, num_classes, tr_dl, val_dl = get_dl(args.root, args.batch_size, valid_transformations)
    print(f"Number of classes in the dataset: {num_classes}\n")
    
    # Initialize model, loss_function, and optimizer    
    model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Set initial best accuracy
    best_accuracy = 0.
    
    # Train model
    train(model, tr_dl, val_dl, num_classes, criterion, optimizer, args.device, args.epochs, best_accuracy)   
    
if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description = 'Object Classification Training Arguments')
    
    # Add arguments
    parser.add_argument("-r", "--root", type = str, default = 'path/to/dataset', help = "Path to the data")
    parser.add_argument("-bs", "--batch_size", type = int, default = 64, help = "Mini-batch size")
    parser.add_argument("-mn", "--model_name", type = str, default = 'rexnet_150', help = "Model name for backbone")
    parser.add_argument("-d", "--device", type = str, default = 'cuda:3', help = "GPU device number")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-3, help = "Learning rate value")
    parser.add_argument("-e", "--epochs", type = int, default = 50, help = "Train epochs number")
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the code
    run(args) 
