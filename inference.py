# Import libraries
import torch, timm

def inference(model_name, num_classes, checkpoint_path, device, dl):
    
    """
    
    This function gets a model name, number of classes for the dataset, path to the trained model, device type, and dataloader;
    performs inference and returns model, predictions, target labels, and images.
    
    Parameters:
    
        model_name      - model name for training, str;
        num_classes     - number of classes for the dataset, int;
        checkpoint_path - path to the trained model, str;
        device          - device type, str;
        dl              - dataloader, torch dataloader instance.
        
    Outputs:
    
        model           - a model for inference, model;
        predictions     - predictions made by the model, tensor;
        gts             - ground truths, tensor;
        images          - images, tensor
        
    """
    
    # Create lists for predictions, ground truths, and images
    predictions, gts, images = [], [], []
    
    # Create a model
    model = timm.create_model(model_name, pretrained = True, num_classes = num_classes)
    
    # Move the model to gpu
    model.to(device)
    
    # Load checkpoint from the path
    model.load_state_dict(torch.load(checkpoint_path))
    print("Model checkpoint loaded successfully!")
    
    # Set initial correct cases and total samples
    correct, total = 0, 0
    
    # Go through the dataloader
    for idx, batch in tqdm(enumerate(dl)):
        
        # Get images and gt labels
        ims, lbls = batch
        
        # Get predictions
        preds = model(ims.to(device))
        images.extend(ims.to(device))
        
        # Get classes with max values
        _, predicted = torch.max(preds.data, 1)
        
        # Add to predictions list
        predictions.extend(predicted)
        
        # Add gt to gts list
        gts.extend(lbls.to(device))
        
        # Add batch size to total number of samples
        total += lbls.size(0)
        
        # Get correct predictions
        correct += (predicted == lbls.to(device)).sum().item()        
        
    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')  
    
    # Return model, predictions, ground truths, and images
    return model, torch.stack(predictions), torch.stack(gts), torch.stack(images)
