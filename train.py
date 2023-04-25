# Import libraries
import timm, torch, os
from tqdm import tqdm

def saveModel(model, save_path, best_accuracy, epoch):
    
    '''
    
    This function gets trained model and saves it as best_model.
    
    Parameter:
    
        model - a trained model.
        
    '''
    
    # Create a dir to save the best model
    os.makedirs(f"{save_path}", exist_ok = True)
    
    # Set the path to save the trained model
    path = f"{save_path}/best_model_{epoch}_{best_accuracy:.1f}.pth"
    
    # Save the model
    torch.save(model.state_dict(), path)

def validation(model, criterion, val_dl, device):
    
    '''
    
    This function gets a model, validation dataloader, and device type; and performs validation process and return accuracy over the whole dataloder.
    
    Parameters:
    
        model     - a trained model;
        val_dl    - validation dataloader;
        device    - device type.
        
    Output:
    
        accuracy  - accuracy of the model on the validation set.
    
    '''
    
    # Change to evaluation mode
    model.eval()
    
    # Set the loss, accuracy and total to 0
    running_loss, running_acc, total = 0, 0, 0

    # Conduct validation without gradients
    with torch.no_grad(): 
        
        for i, batch in enumerate(val_dl):

            # Get the data and gt 
            images, labels = batch
            
            # Move them the gpu
            images, labels = images.to(device), labels.to(device)
            # Get the model predictions
            outputs = model(images)
            # Compute loss and add to running loss
            loss = criterion(outputs, labels)
            running_loss += loss.item() 
            
            # Select the prediction with the highest value
            _, predicted = torch.max(outputs.data, 1)
            
            # Add to total number of samples
            total += labels.size(0)
            
            # Get the accuracy
            running_acc += (predicted == labels).sum().item()
    
    # Compute the loss and accuracy over all test images and return the values
    return (running_loss / total), (100 * running_acc / total)
    
def train(model, tr_dl, val_dl, num_classes, criterion, optimizer, device, epochs, best_accuracy, save_path):
    
    '''
    
    This function gets a model, train dataloader, validation dataloader, optimizer, 
    loss_function, number of epochs, and device type and trains the model.
    
    Parameters:
    
        model         - a trained model;
        tr_dl         - train dataloader;
        val_dl        - validation dataloader;
        num_classes   - number of classes;
        criterion     - loss function;
        optimizer     - optimizer type;
        device        - device type;
        epochs        - number of epoch to train the model;
        best_accuracy - current best accuracy, default 0.
    
    '''

    # Define your execution device
    print(f"The model will be running on {device} device\n")
    
    # Initialize lists to track metrics
    tr_loss, val_loss, tr_accs, val_accs = [], [], [], []
    
    # Move the model to gpu
    model.to(device)
    
    # Start training
    for epoch in range(epochs): 
        
        # Set running loss and accuracy
        running_loss, running_acc, total = 0, 0, 0
        
        # Get through the training dataloader
        for i, batch in tqdm(enumerate(tr_dl, 0)):
            
            # Get the inputs and move them to device
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            total += labels.shape[0]

            # Zero the parameter gradients
            optimizer.zero_grad()
            # Predict classes using images from the training dataloader
            outputs = model(images)
            
            # Get the accuracy
            _, predicted = torch.max(outputs.data, 1)
            running_acc += (predicted == labels).sum().item()
            
            # Compute the loss based on model output and real labels
            loss = criterion(outputs, labels)
            running_loss += loss.item()     # extract the loss value
            
            # Backpropagate the loss
            loss.backward()
            # Adjust parameters based on the calculated gradients
            optimizer.step()

        # Add train loss and accuracy to the lists
        tr_loss.append(running_loss / total)
        tr_accs.append(running_acc / total)
        # Perform validation and get the accuracy score
        val_loss_, val_accuracy = validation(model, criterion, val_dl, device)
        print(f"\nValidation accuracy on epoch {epoch+1} is {val_accuracy:.3f}%")
        print(f"Validation loss on epoch {epoch+1} is {val_loss_:.3f}\n")
        
        # Add validation loss and accuracy to the lists
        val_loss.append(val_loss_)
        val_accs.append(val_accuracy)
        
        # Save the model with the best accuracy
        print(f"The best validation accuracy on epoch {epoch} is {best_accuracy:.2f}%")
        if val_accuracy > best_accuracy:
            
            # Change best accuracy value
            best_accuracy = val_accuracy
            # Save the model with the best accuracy
            saveModel(model, save_path, best_accuracy, epoch+1)
            print(f"The best validation accuracy on epoch {epoch+1} is {best_accuracy:.2f}%")
            
    # Return dictionary with train results
    return {"tr_loss": tr_loss, "tr_accs": tr_accs, "val_loss": val_loss, "val_accs": val_accs}
