import timm, torch
from tqdm import tqdm

def saveModel(model):
    
    path = "./best_model.pth"
    torch.save(model.state_dict(), path)

def validation(model, val_dl, device):
    
    model.eval()
    accuracy, total = 0, 0

    with torch.no_grad():
        for data in val_dl:
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
    
    return accuracy
    
def train(model_name, tr_dl, val_dl, num_classes, lr, device, epochs):

    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    best_accuracy = 0.0

    # Define your execution device
    print(f"The model will be running on {device} device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

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

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            if i % 1000 == 999:    
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = validation(model, val_dl, device)
        print(f"For epoch {epoch+1} the test accuracy over the whole test set is {accuracy:.3f}%")
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel(model)
            best_accuracy = accuracy
    
