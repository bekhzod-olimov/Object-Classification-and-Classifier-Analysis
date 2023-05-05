# Import libraries
import os, torch, pickle, timm, gdown, argparse, gradio as gr, numpy as np
from transforms import get_transforms 
from glob import glob
from PIL import Image, ImageFont
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def load_model(model_name, num_classes, checkpoint_path): 
    
    """
    
    This function gets several parameters and loads a classification model.
    
    Parameters:
    
        model_name      - name of a model from timm library, str;
        num_classes     - number of classes in the dataset, int;
        checkpoint_path - path to the trained model, str;
        
    Output:
    
        m               - a model with pretrained weights and in an evaluation mode, torch model object;
    
    """
    
    # Download from the checkpoint path
    if os.path.isfile(checkpoint_path): print("Pretrained model is already downloaded!"); pass
    
    # If the checkpoint does not exist
    else: 
        print("Pretrained checkpoint is not found!")
        
        # Set url path
        url = "https://drive.google.com/file/d/1T6joFbxQN1aWesmCOWAn07t8kmoabIH8/view?usp=share_link"
        
        # Get file id
        file_id = url.split("/")[-2]
        
        # Initialize prefix to download
        prefix = "https://drive.google.com/uc?/export=download&id="
        
        # Download the checkpoint
        gdown.download(prefix + file_id, checkpoint_path, quiet = False)
    
    # Create a model based on the model name and number of classes
    m = timm.create_model(model_name, num_classes = num_classes)
    
    # Load the state dictionary from the checkpoint
    m.load_state_dict(torch.load(checkpoint_path, map_location = "cpu"))
    
    # Switch the model into evaluation mode
    return m.eval()

def run(args):
    
    """
    
    This function gets parsed arguments and runs the script.
    
    Parameter:
    
        args   - parsed arguments, argparser object;
        
    """
    
    # Get class names for later use
    with open("cls_names.pkl", "rb") as f: cls_names = pickle.load(f)
    
    # Get number of classes
    num_classes = len(cls_names)
    
    # Initialize transformations to be applied
    tfs = get_transforms(train = False)
    
    title = "Online Object Classifier"
    
    # Set the description
    desc = "The model is just  trained using only 30 objects and serves only for demonstration purposes. It should be trained using more data to show better classificiation results. \n Please choose one of the images listed below or upload your own image using 'Click to Upload' and see the classification result!"
    
    # Get the samples to be classified
    examples = [[im] for im in glob(f"{args.root}/*/*.jpg")]
    
    # Initialize inputs with label
    inputs = gr.inputs.Image(label = "Object to be Classified")
    
    # Get the model to classify the objects
    model = load_model(args.model_name, num_classes, args.checkpoint_path)

    def predict(inp):
    
        im = tfs(Image.fromarray(inp.astype('uint8'), 'RGB'))
        cam = GradCAM(model = model, target_layers = [model.features[-1]], use_cuda = False)
        grayscale_cam = cam(input_tensor = im.unsqueeze(0).to("cpu"))[0, :]
        visualization = show_cam_on_image((im*255).cpu().numpy().transpose([1,2,0]).astype(np.uint8)/255, grayscale_cam, image_weight = 0.55, colormap = 2, use_rgb = True)
        
        return Image.fromarray(visualization), cls_names[int(torch.max(model(im.unsqueeze(0)).data, 1)[1])]
    
    outputs = [gr.outputs.Image(type = "numpy", label = "GradCAM Result"), gr.outputs.Label(type = "numpy", label = "Predicted Count")]
    gr.Interface(fn = predict, inputs = inputs, outputs = outputs, title = title, description = desc, examples = examples, allow_flagging=False).launch(share = True)

if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description = 'Object Classification Demo')
    
    # Add arguments
    parser.add_argument("-r", "--root", type = str, default = "/home/ubuntu/workspace/bekhzod/Object-Classification-and-Classifier-Analysis/", help = "Root for sample images")
    parser.add_argument("-mn", "--model_name", type = str, default = 'rexnet_150', help = "Model name for backbone")
    parser.add_argument("-cp", "--checkpoint_path", type = str, default = 'saved_models/best_model_11_98.6.pth', help = "Path to the checkpoint")
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the code
    run(args)
