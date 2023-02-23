import torch, argparse, yaml, timm
from transforms import get_transforms
from dataset import get_dl
from train import train
from tqdm import tqdm

def run(args):
    
    # Get train arguments    
    root = args.root
    bs = args.batch_size
    device = args.device
    lr = args.learning_rate
    model_name = args.model_name
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print(f"\nTraining Arguments:\n{argstr}")
    
    # Set train variable
    epochs = 50
    
    # Get train and validation transformations 
    train_transformations, valid_transformations= get_transforms(train=True), get_transforms(train=False)
    
    # Get class names, number of classes, train and validation dataloaders
    cls_names, num_classes, tr_dl, val_dl = get_dl(root, bs, valid_transformations)
    print(f"Number of classes in the dataset: {num_classes}\n")
    
    # Initialize model, loss_function, and optimizer    
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Set initial best accuracy
    best_accuracy = 0.
    
    # Train model
    train(model, tr_dl, val_dl, num_classes, criterion, optimizer, device, epochs, best_accuracy)   
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'Object Classification Training Arguments')
    parser.add_argument("-r", "--root", type = str, default = 'simple_classification', help = "Path to the data")
    parser.add_argument("-bs", "--batch_size", type = int, default = 64, help = "Mini-batch size")
    parser.add_argument("-mn", "--model_name", type = str, default = 'rexnet_150', help = "Model name for backbone")
    parser.add_argument("-d", "--device", type = str, default = 'cuda:3', help = "GPU device number")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-3, help = "Learning rate value")
    args = parser.parse_args() 
    
    run(args) 
