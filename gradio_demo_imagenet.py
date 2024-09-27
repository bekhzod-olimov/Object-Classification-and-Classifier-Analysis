# Import libraries
import os, torch, pickle, timm, gdown, argparse, gradio as gr, numpy as np
from transforms import get_transforms; from glob import glob; from torchvision import transforms as T
from PIL import Image, ImageFont; from torchvision.datasets import ImageFolder
from pytorch_grad_cam import GradCAM; from pytorch_grad_cam.utils.image import show_cam_on_image

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
    
    # If the pretrained file is in the local path
    if os.path.isfile(checkpoint_path): print("Pretrained model is already downloaded!"); pass
    
    # If the pretrained file has not been downloaded yet
    else: 
        print("Pretrained checkpoint is not found!")
        # Url of the checkpoint
        url = "https://drive.google.com/file/d/1T6joFbxQN1aWesmCOWAn07t8kmoabIH8/view?usp=share_link"
        # Get file id
        file_id = url.split("/")[-2]
        # Set prefix
        prefix = "https://drive.google.com/uc?/export=download&id="
        # Download the checkpoint
        gdown.download(prefix + file_id, checkpoint_path, quiet = False)
     
    m = timm.create_model(model_name, pretrained = True, num_classes = num_classes)
    
    return m.eval()

def run(args):
    
    """
    
    This function gets parsed arguments and runs the script.
    
    Parameter:
    
        args   - parsed arguments, argparser object;
        
    """
    
    # Get class names for later use
    cls_names = dict(enumerate(open("imagenet_classes.txt")))
    
    # Get number of classes
    num_classes = len(cls_names)
    
    # Initialize transformations to be applied
    tfs = get_transforms(train = False)
    
    # Set the title
    title = "Online Object Classifier"
    
    # Set the description
    desc = "Please choose one of the images listed below or upload your own image using 'Click to Upload' and see the classification result!"
    
    # Get sample images from the folder
    examples = [[im] for im in glob(f"{args.root}/*")]
    
    # Initialize inputs
    inputs = gr.inputs.Image(label = "Object to be Classified")
    
    # Load pretrained model
    model = load_model(args.model_name, num_classes, args.checkpoint_path)

    def predict(inp):
        
        """
        
        This function gets an input image and predicts its class.
        
        Parameter:
        
            inp    - an input image, array.
            
        Output:
        
            im     - a GradCAM applied output image, array;
            out    - predicted class name, str.
        
        """
    
        # Get an image to be classified
        im = tfs(Image.fromarray(inp.astype("uint8"), "RGB"))
        
        # Initialize GradCAM object
        cam = GradCAM(model = model, target_layers = [model.features[-1]], use_cuda = False)
        
        # Get grayscale GradCAM image
        grayscale_cam = cam(input_tensor = im.unsqueeze(0).to("cpu"))[0, :]
        
        # Get GradCAM output
        visualization = show_cam_on_image((im*255).cpu().numpy().transpose([1,2,0]).astype(np.uint8)/255, grayscale_cam, image_weight = 0.55, colormap = 2, use_rgb = True)
        
        return Image.fromarray(visualization), cls_names[int(torch.max(model(im.unsqueeze(0)).data, 1)[1].item())]
    
    # Initialize outputs list
    outputs = [gr.outputs.Image(type = "numpy", label = "GradCAM Result"), gr.outputs.Label(type = "numpy", label = "Predicted Label")]
    
    # Set gradio interface
    gr.Interface(fn = predict, inputs = inputs, outputs = outputs, title = title, description = desc, examples = examples, allow_flagging = False).launch(share = True)

if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description = "Object Classification Demo")
    
    # Add arguments
    parser.add_argument("-r", "--root", type = str, default = "path/to/data", help = "Root for sample images")
    parser.add_argument("-mn", "--model_name", type = str, default = "rexnet_150", help = "Model name for backbone")
    parser.add_argument("-cp", "--checkpoint_path", type = str, default = "path/to/checkpoint", help = "Path to the checkpoint")
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the code
    run(args)
