# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torch.utils.data 

import json
from PIL import Image
from os import listdir
import argparse

# Set up arguments for the command line
parser = argparse.ArgumentParser(description='Set Up Neural Networks')
parser.add_argument('image_path', type=str, help='Enter the path of an image to predict.')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='Enter checkpoint to load the trained model.')
parser.add_argument('--top_k', type=int, default=5, help='Enter number of top most likely classes.')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Enter json file holding category names.')
parser.add_argument('--gpu', action='store_true', help='Use GPU.')

args = parser.parse_args()

image_path = args.image_path

if args.image_path:
    image_path = args.image_path
if args.checkpoint:
    checkpoint = args.checkpoint
if args.top_k:
    top_k = args.top_k
if args.category_names:
    category_names = args.category_names
if args.gpu:        
    device = 'cuda'
else:
    device = 'cpu'
    
# Load the mapping of category names
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

# Load the checkpoint
def load_checkpoint(checkpoint):
    """
    Use checkpoint to load the trained model.
    """
    checkpoint = torch.load(checkpoint)
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    else: 
        model = models.alexnet(pretrained=True)
    
    model.classifier = checkpoint['classifier']  
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping']
    
    for param in model.parameters():
        param.requires_grad=False
    
    return model

model = load_checkpoint(checkpoint)

print('-'*10)
print('Your trained model has been loaded. Predicting the image......')
print('-'*10)

# Process the image
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    ''' 
    # Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)
    
    # Resize the image
    size = 256
    width, height = image.size
    aspect = float(width) / float(height)
    
    if width > height:
        new_height = size
        new_width = int(aspect * size)
    else:
        new_width = size
        new_height = int(size/aspect)
    
    image.thumbnail((new_width, new_height), Image.ANTIALIAS)
    
    # Crop the center of the image
    crop_size = 224
    
    left = (new_width - crop_size)/2
    top = (new_height - crop_size)/2
    right = left + crop_size
    bottom = top + crop_size

    image = image.crop((left, top, right, bottom))
    
    # Convert color channels to 0-1
    np_image = np.array(image)/255
    
    # Normalize the image
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    np_image = (np_image - means) / stds
    
    # Reorder dimensions
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

# Predict the image
def predict(image_path, model, top_k):
    """
    Use image path, trained model and top k to predict the category names and probabilities of an image.
    """
    np_image = process_image(image_path)
    
    # Convert numpy array to tensor
    image = torch.from_numpy(np_image).type(torch.FloatTensor)
    image = image.unsqueeze (dim=0)
    
    # Use model to predict
    model.to(device)
    image.to(device)
    
    model.eval()
    with torch.no_grad():
        log_probs = model(image)
    
    probs = torch.exp(log_probs)
    probs, indices = probs.topk(top_k, dim=1)
    
    # Convert probs, indices to list
    probs, indices = probs.to('cpu'), indices.to('cpu')
    probs, indices = probs.tolist()[0], indices.tolist()[0]
    
    # Invert class_to_idx to idx_to_class
    class_to_idx = model.class_to_idx
        
    idx_to_class = {i: class_ for class_, i in class_to_idx.items()}    
    
    # Convert index to class  
    classes = []
    for i in indices:
        classes.append(idx_to_class[i])
    
    # Convert class to name
    names = []
    for class_ in classes:
        names.append(cat_to_name[class_])
    
    result_dict = {'flower_name': pd.Series(names), 'prob': pd.Series(probs)}
    result = pd.DataFrame(result_dict).set_index('flower_name')
    
    return result

result = predict(image_path, model, top_k)
print(result)