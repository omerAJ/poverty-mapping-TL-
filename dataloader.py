import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import torch
from PIL import Image
import torchvision.transforms.functional as TF

### class_counts = {0: 307, 1: 1475, 2: 610}

class customDataset(Dataset):
    def __init__(self, directory, transform=None, num_crops=1):
        """
        Args:
            directory (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.num_crops = num_crops
        self.directory = directory
        self.transform = transform
        self.images = [f for f in os.listdir(directory) if f.endswith('.tif')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.images[idx])
        # image = read_image(img_name) # Loads image as a tensor  use if png, jpg
        image = Image.open(img_name)  # PIL.Image.open supports TIFF
        image = TF.to_tensor(image)  # Convert PIL image to PyTorch tensor

        class_label = int(self.images[idx].split(',')[0].strip('('))
        
        if self.transform:
            image = self.transform(image)
        
        return image, class_label

def create_dataloader(directory, batch_size, transform=None):
    dataset = Dataset(directory=directory, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import defaultdict
import os

def get_sample_weights(directory, class_counts):
    # Initialize a list to hold the weights for each sample
    sample_weights = []
    
    # List all files in the given directory
    files = [f for f in os.listdir(directory) if f.endswith('.tif')]
    
    # Calculate the total number of samples
    n_samples = len(files)
    
    # Calculate the weight for each class (inverse frequency)
    weights_per_class = {class_id: n_samples / class_counts[class_id] for class_id in class_counts}
    
    # Loop through each file, determine its class, and assign a weight based on its class
    for file_name in files:
        class_label = int(file_name.split(',')[0].strip('('))
        weight = weights_per_class[class_label]
        sample_weights.append(weight)
    
    return sample_weights
