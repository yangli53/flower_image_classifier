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
parser.add_argument('data_dir', type=str, help='Enter path to load the data.')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Enter path to save the checkpoint.')
parser.add_argument('--arch', type=str, default='vgg16', help='Enter a pretrained model to use, either "vgg16" or "alexnet".')
parser.add_argument('--hidden_units', type=int, default=1000, help='Enter number of hidden units for training.')                  
parser.add_argument('--learning_rate', type=float, default=0.001, help='Enter learning rate for the training model.')
parser.add_argument('--epochs', type=int, default=10, help='Enter number of epochs for training.')
parser.add_argument('--gpu', action='store_true', help='Use GPU.')

args = parser.parse_args()

if args.arch:
    arch = args.arch
if args.hidden_units:
    hidden_units = args.hidden_units
if args.learning_rate:
    learning_rate = args.learning_rate
if args.epochs:
    epochs = args.epochs
if args.gpu:        
    device = 'cuda'
else:
    device = 'cpu'

# Set up data loading
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
    
# Load the data
# Define transforms
data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30), 
                                                transforms.RandomResizedCrop(224), 
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(), 
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])
                                               ]),
                   'valid': transforms.Compose([transforms.Resize(255), 
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(), 
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])
                                               ]),
                   'test': transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(), 
                                               transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])
                                              ])
                  }

# Load the datasets with ImageFolder
image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                  'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                  'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
                 }

# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
               'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
               'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
              }

print('-'*10)
print('Your data has been loaded.')
print('-'*10)
    
# Build a training model
def build_model(arch, hidden_units, learning_rate):
    """
    Set up arch, hidden_units and learning_rate to build a training model.
    """
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = 25088
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        in_features = 9216
    else:
        arch == 'vgg16'
        model = models.vgg16(pretrained=True)
        in_features = 25088
        print('This script can only apply "vgg16" or "alexnet". Default model "vgg16" has been applied.')
        
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Define a new, untrained feed-forward network as a classifier
    classifier = nn.Sequential(nn.Linear(in_features, 4096), 
                               nn.ReLU(), 
                               nn.Dropout(0.2), 
                               nn.Linear(4096, hidden_units),
                               nn.ReLU(), 
                               nn.Dropout(0.2), 
                               nn.Linear(hidden_units, 102), 
                               nn.LogSoftmax(dim=1))
    
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer

model, criterion, optimizer = build_model(arch, hidden_units, learning_rate)

print('-'*10)
print("Your model has been built.")
print('-'*10)

# Define validation 
def validation(model, dataloader, criterion):  
    """
    Use model, dataloader and criterion to calculate validation loss and accuracy.
    """
    model.to(device)
    
    loss = 0
    accuracy = 0
            
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            log_probs = model(images)
            batch_loss = criterion(log_probs, labels)
                    
            loss += batch_loss.item()
                    
            probs = torch.exp(log_probs)
            top_prob, top_class = probs.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            batch_accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
                    
            accuracy += batch_accuracy
                    
    return loss, accuracy

# Train the model
def train_model(model, criterion, optimizer, epochs):
    """
    Set up model, criterion, optimizer and epochs to train the model.
    """
    model.to(device)

    steps = 0
    running_loss = 0
    print_every = 50

    for epoch in range(epochs):
        for images, labels in dataloaders['train']:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            log_probs = model(images)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                valid_loss, accuracy = validation(model, dataloaders['valid'], criterion)

                valid_len = len(dataloaders["valid"])

                print(f'Epoch: {epoch+1}/{epochs}..'
                      f'Train loss: {running_loss/print_every:.3f}..'
                      f'Valid loss: {valid_loss/valid_len:.3f}..'
                      f'Valid accuracy: {accuracy*100/valid_len:.3f}%')

                running_loss = 0  
                model.train()
                
    return model

trained_model = train_model(model, criterion, optimizer, epochs)

print('-'*10)
print('Your model has been trained')
print('-'*10)

# Save the checkpoint
if args.save_dir:
    save_dir = args.save_dir
    
def save_model(trained_model):
    """
    Use checkpoint to save the trained model.
    """
    trained_model.to('cpu')

    trained_model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoints = {'arch': arch,
                   'classifier': trained_model.classifier,
                   'state_dict': trained_model.state_dict(),
                   'mapping': trained_model.class_to_idx}

    torch.save(checkpoints, save_dir)
    
save_model(trained_model)
print('-'*10)
print('Your model has been saved')
print('-'*10)