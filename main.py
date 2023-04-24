# Import libraries
import torch, argparse, yaml, timm, gdown, pickle, os
from transforms import get_transforms
from dataset import get_dl
from train import train
from tqdm import tqdm

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
    
    # Get train and validation transformations 
    train_transformations, valid_transformations= get_transforms(train = True), get_transforms(train = False)
    
    # Get class names, number of classes, train and validation dataloaders
    cls_names, num_classes, tr_dl, val_dl = get_dl(args.root, args.batch_size, valid_transformations)
    print(f"Number of classes in the dataset: {num_classes}\n")
    
    # Initialize model, loss_function, and optimizer    
    model = timm.create_model(args.model_name, pretrained = True, num_classes = num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.learning_rate)
    
    # Set initial best accuracy
    best_accuracy = 0.
    
    # Train model
    results = train(model, tr_dl, val_dl, num_classes, criterion, optimizer, args.device, args.epochs, best_accuracy, args.save_model_path)   
    
    # Save the dictionary
    os.makedirs(f"{args.save_results_path}", exist_ok = True)
    with open(f"{args.save_results_path}/{args.model_name}_{args.batch_size}_{args.epochs}_{args.learning_rate}_results.pkl", 'wb') as file:
        pickle.dump(results, file)
        
    # Load the dictionary
    # with open(f"{args.save_results_path}/{args.model_name}_{args.bs}_{args.epochs}_{args.learning_rate}_results.pkl", 'rb') as file:
    #     results_dict = pickle.load(file)
    
if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description = 'Object Classification Training Arguments')
    
    # Add arguments
    parser.add_argument("-r", "--root", type = str, default = 'path/to/data', help = "Path to the data")
    parser.add_argument("-bs", "--batch_size", type = int, default = 128, help = "Mini-batch size")
    parser.add_argument("-mn", "--model_name", type = str, default = 'rexnet_150', help = "Model name for backbone")
    parser.add_argument("-d", "--device", type = str, default = 'cuda:3', help = "GPU device number")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-3, help = "Learning rate value")
    parser.add_argument("-e", "--epochs", type = int, default = 50, help = "Train epochs number")
    parser.add_argument("-sm", "--save_model_path", type = str, default = 'saved_models', help = "Path to the directory to save a trained model")
    parser.add_argument("-sr", "--save_results_path", type = str, default = 'results', help = "Path to the directory to save the train results")
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the code
    run(args) 
