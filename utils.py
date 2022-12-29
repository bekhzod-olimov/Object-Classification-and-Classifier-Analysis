import timm, torch
from tqdm import tqdm

def saveModel(model):
    
    path = "./best_model.pth"
    torch.save(model.state_dict(), path)

def validation(model, val_dl, device):
    
    model.eval()
    accuracy, total = 0, 0

    with torch.no_grad():
        
        for i, data in enumerate(val_dl):

            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    print(f"Validation accuracy is: {accuracy:.3f}")
    
    return accuracy
    
def train(model, tr_dl, val_dl, num_classes, criterion, optimizer, device, epochs, best_accuracy):

    # Define your execution device
    print(f"The model will be running on {device} device")
    model.to(device)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        running_loss, running_acc = 0, 0
        
        for i, (images, labels) in tqdm(enumerate(tr_dl, 0)):
            
            # get the inputs
            images = images.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = criterion(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()
            running_loss += loss.item()     # extract the loss value

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = validation(model, val_dl, device)
        print(f"For epoch {epoch+1} the validation accuracy over the whole validation set is {accuracy:.2f}%")
        
        # we want to save the model if the accuracy is the best
        print(f"The best validation accuracy on epoch {epoch+1} is {best_accuracy:.2f}%")
        if accuracy > best_accuracy:
            saveModel(model)
            best_accuracy = accuracy
            print(f"The best validation accuracy on epoch {epoch+1} is {best_accuracy:.2f}%")
    
