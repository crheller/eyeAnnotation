import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
import json
from PIL import Image

class CustomCNN(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block (60x60 -> 30x30)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Second conv block (30x30 -> 15x15)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Third conv block (15x15 -> 7x7)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Fourth conv block (7x7 -> 3x3)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),

            # Fifth conv block (7x7 -> 3x3)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer for 9 regression targets
            nn.Linear(128, 9)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

# ================================== Deeper CNN structure =============================   

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection with 1x1 conv if dimensions change
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv_block(x) + self.skip(x))


class DeepCNN(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(DeepCNN, self).__init__()
        
        # Initial convolution
        self.input_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 60x60
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        
        # First stage - no spatial reduction
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.Dropout2d(dropout_rate)
        )  # Still 60x60
        
        # Second stage - reduce to 30x30
        self.stage2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),  # Spatial reduction here
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.Dropout2d(dropout_rate)
        )  # Now 30x30
        
        # Third stage - reduce to 15x15
        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.Dropout2d(dropout_rate)
        )  # Now 15x15
        
        # Fourth stage - reduce to 8x8
        self.stage4 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.Dropout2d(dropout_rate)
        )  # Now 8x8
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers with skip connections
        self.fc1 = nn.Linear(256, 512)
        self.fc1_skip = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc2_skip = nn.Linear(512, 256)
        
        # Output layer
        self.output = nn.Linear(256, 9)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Convolutional stages
        x = self.input_block(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with skip connections
        identity = x
        x = self.fc1(x)
        x = x + self.fc1_skip(identity)
        x = self.relu(x)
        x = self.dropout(x)
        
        identity = x
        x = self.fc2(x)
        x = x + self.fc2_skip(identity)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.output(x)
        return x
    

# ================================= DATA LOADER ===================================
class EyeAngleDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, scale_output=None):
        """
        Args:
            image_dir (string): Directory with all the images
            label_file (string): Path to the json file with annotations
            transform (callable, optional): Optional transform to be applied on a sample
            scale_output: np array of length n_outputs that scales each accordingly
        """
        self.image_dir = image_dir
        self.label_dir = label_dir 
        self.transform = transform
        self.scale_output = scale_output
        self.labels = os.listdir(label_dir)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        fname = self.labels[idx].strip(".json")
        img_name = os.path.join(self.image_dir, f"{fname}.png")
        # Load image (assuming grayscale)
        image = Image.open(img_name).convert("L")
        with open(os.path.join(self.label_dir, f"{fname}.json"), "r") as f:
            _annotations = json.load(f)[0]

        labels = np.array([
            _annotations["left_eye_x_position"],
            _annotations["left_eye_y_position"],
            _annotations["right_eye_x_position"],
            _annotations["right_eye_y_position"],
            _annotations["yolk_x_position"],
            _annotations["yolk_y_position"],
            _annotations["left_eye_angle"],
            _annotations["right_eye_angle"],
            _annotations["heading_angle"]
        ]).astype(np.float32)
        
        if self.transform:
            image = self.transform(image)
        if self.scale_output is not None:
            labels *= self.scale_output
                   
        return image, labels