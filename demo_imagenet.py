# Import libraries
import torch, pickle, timm, argparse, streamlit as st
from transforms import get_transforms  
from PIL import Image, ImageFont
from torchvision.datasets import ImageFolder
st.set_page_config(layout='wide')

def run(args):
    
    """
    
    This function gets parsed arguments and runs the script.
    
    Parameter:
    
        args   - parsed arguments, argparser object;
        
    """
    
    # Get class names for later use
    with open('cls_names.pkl', 'rb') as f: cls_names = pickle.load(f)
    
    # Get number of classes
    num_classes = len(cls_names)
    
    # Initialize transformations to be applied
    tfs = get_transforms(train = False)
    
    # Set a default path to the image
    default_path = "airpods.jpg"
    
#     ds = ImageFolder(root = "/home/ubuntu/workspace/bekhzod/triplet-loss-pytorch/pytorch_lightning/data/simple_classification", transform = tfs)
    
#     # Get class names
#     cls_names = list(ds.class_to_idx.keys())
    
#     with open('cls_names.pkl', 'wb') as f: 
#         pickle.dump(cls_names, f)
    
    # Load classification model
    m = load_model(args.model_name, num_classes, args.checkpoint_path)
    st.title("Object Recognition")
    file = st.file_uploader('Please upload your image')

    # Get image and predicted class
    im, out = predict(m = m, path = file, tfs = tfs, cls_names = cls_names) if file else predict(m = m, path = default_path, tfs = tfs, cls_names = cls_names)
    st.write(f"Input Image: ")
    st.image(im)
    st.write(f"Predicted as {out}")
        
@st.cache_data
def load_model(model_name, num_classes, checkpoint_path): 
    
    """
    
    This function gets several parameters and loads a classification model.
    
    Parameters:
    
        model_name      - name of a model from timm library, str;
        num_classes     - number of classes in the dataset, int;
        checkpoint_path - path to the trained model, str;
        
    Output:
    
        m              - a model with pretrained weights and in an evaluation mode, torch model object;
    
    """
    
    m = timm.create_model(model_name, num_classes = num_classes)
    m.load_state_dict(torch.load(checkpoint_path, map_location = "cpu"))
    
    return m.eval()

def predict(m, path, tfs, cls_names):
    
    fontpath = "SpoqaHanSansNeo-Light.ttf"
    font = ImageFont.truetype(fontpath, 200)
    im = Image.open(path)
    im.save(path)
    
    return im, cls_names[int(torch.max(m(tfs(im).unsqueeze(0)).data, 1)[1])]
        
if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description = 'Object Classification Demo')
    
    # Add arguments
    parser.add_argument("-mn", "--model_name", type = str, default = 'rexnet_150', help = "Model name for backbone")
    parser.add_argument("-cp", "--checkpoint_path", type = str, default = 'saved_models/best_model_11_98.6.pth', help = "Path to the checkpoint")
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the code
    run(args) 
