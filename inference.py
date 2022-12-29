import torch, timm

def inference(model_name, num_classes, checkpoint_path, device, dl):
    
    '''
    Gets a model name, number of classes for the dataset, path to the trained model, device type, and dataloader;
    performs inference and returns model, predictions, target labels, and images.
    
    Arguments:
    model_name - model name for training;
    num_classes - number of classes for the dataset;
    checkpoint_path - path to the trained model;
    device - device type;
    dl - dataloader.
    '''
    
    predictions, gts, images = [], [], []
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    print("Model checkpoint loaded successfully!")
    
    correct, total = 0, 0
    for idx, batch in tqdm(enumerate(dl)):
        ims, lbls = batch
        preds = model(ims.to(device))
        images.extend(ims.to(device))
        _, predicted = torch.max(preds.data, 1)
        predictions.extend(predicted)
        gts.extend(lbls.to(device))
        total += lbls.size(0)
        correct += (predicted == lbls.to(device)).sum().item()        
        
    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')  
    
    return model, torch.stack(predictions), torch.stack(gts), torch.stack(images)
