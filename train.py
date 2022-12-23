import torch, os, argparse, yaml
from transforms import get_transforms
from dataset import get_dl
from utils import train

def run(args):
    
    root = args.root
    bs = args.batch_size
    device = args.device
    lr = args.learning_rate
    model_name = args.model_name
    epochs = 50
    
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print(f"\nTraining Arguments:\n{argstr}\n")
    
    train_transformations, valid_transformations= get_transforms(train=True), get_transforms(train=False)
    cls_names, num_classes, tr_dl, val_dl = get_dl(root, bs, valid_transformations)
    train(model_name, tr_dl, val_dl, num_classes, lr, device, epochs)   
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Object Classification Training Arguments')
    parser.add_argument("-r", "--root", type=str, default='simple_classification', help="Path to the data")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-mn", "--model_name", type=str, default='rexnet_150', help="Model name for backbone")
    parser.add_argument("-d", "--device", type=str, default='cuda:3', help="GPU device number")
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-3, help="Learning rate value") # from find_lr
    args = parser.parse_args() 
    
    run(args) 