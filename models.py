import torch
import torch.nn as nn
from torchvision import models

def load_modified_resnet50(num_classes):
    # Load a pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)
    
    # Get the input dimension of the original fully connected layer
    num_ftrs = model.fc.in_features
    
    # Replace the final fully connected layer with a new one with the desired number of output classes
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

    