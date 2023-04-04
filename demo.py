# Import libraries
import torch, pickle, timm, argparse
import streamlit as st
from transforms import get_transforms  
from PIL import Image, ImageFont
from torchvision.datasets import ImageFolder
st.set_page_config(layout='wide')

def run(args):
    
    # This is demo
    
    """
    
    This function gets parsed arguments and runs the script.
    
    Argument:
    
        args - parsed arguments, parser object.
    
    """
    
    # Get class names list
    with open('cls_names.pkl', 'rb') as f: cls_names = pickle.load(f)
    
    # Initialize data transformations 
    tfs = get_transforms(train = False)
    
    # Get number of classes in the dataset
    num_classes = len(cls_names)
    
#     ds = ImageFolder(root = "/home/ubuntu/workspace/bekhzod/triplet-loss-pytorch/pytorch_lightning/data/simple_classification", transform = tfs)
    
#     # Get class names
#     cls_names = list(ds.class_to_idx.keys())
    
#     with open('cls_names.pkl', 'wb') as f: 
#         pickle.dump(cls_names, f)
    
    # Load pretrained model 
    m = load_model(args.model_name, num_classes, args.checkpoint_path)
    
    # Write title 
    st.title("Object Recognition")
    
    # Initialize file (image) loader
    file = st.file_uploader('Please upload your image')
    
    # Once the file is uploaded
    if file:
        
        # Get an image and predicted class 
        im, out = predict(m, file, tfs, cls_names)

        # Write descriptions
        st.write(f"Input Image: ")
        
        # Visualize the image
        st.image(im)
        
        # Write the predicted class name
        st.write(f"Predicted as {out}")
        
@st.cache_data
def load_model(model_name, num_classes, checkpoint_path): 
    
    m = timm.create_model(model_name, num_classes = num_classes)
    m.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
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
