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

    
def load_modified_resnet50_for_regression(num_classes, state_dict_path=None, freeze_weights=False):
    # Load a pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)
    
    # Get the input dimension of the original fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    # Load custom state_dict if specified
    if state_dict_path is not None:
        custom_state_dict = torch.load(state_dict_path)
        model.load_state_dict(custom_state_dict)
    
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    return model

def load_resnet50_for_regression(state_dict_path=None, freeze_weights=False):
    # Load a pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )

    if state_dict_path is not None:
        custom_state_dict = torch.load(state_dict_path)
        model.load_state_dict(custom_state_dict)

    model.fc = torch.nn.Identity()  ## removing the regression head.
    
    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = False

    return model