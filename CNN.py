import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
import json
from PIL import Image
import math
from torchvision import models

class ResNet50Regression(nn.Module):
    def __init__(self, num_outputs=9, pretrained=True, freeze_backbone=False):
        """
        Initializes the ResNet50 model for regression with a custom output layer.

        Args:
            num_outputs (int): The number of regression output variables.
            pretrained (bool): Whether to load pre-trained weights for ResNet50.
        """
        super(ResNet50Regression, self).__init__()
        
        # Load the pretrained ResNet50 model
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final fully connected layer
        # Replace with a new fully connected layer to output the desired number of regression values
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_outputs)

        # Freeze the backbone if specified
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

            # Unfreeze the last fully connected layer for training
            for param in self.resnet.fc.parameters():
                param.requires_grad = True
        
    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_outputs)
        """
        return self.resnet(x)

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
    
class ShallowCNN(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(ShallowCNN, self).__init__()
    
        # Initial convolution layer
        self.input_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input size: 60x60
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        
        # First stage - reduce to 30x30
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),  # First spatial reduction
            ResidualBlock(64, 64),
            nn.Dropout2d(dropout_rate)
        )  # Now 30x30
        
        # Second stage - reduce to 15x15
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),  # Second spatial reduction
            ResidualBlock(128, 128),
            nn.Dropout2d(dropout_rate)
        )  # Now 15x15
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 
        
        # Fully connected layers with skip connections
        self.fc1 = nn.Linear(128, 64)
        self.fc1_skip = nn.Linear(128, 64)
        
        # Output layer
        self.output = nn.Linear(64, 9)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Convolutional stages
        x = self.input_block(x)
        x = self.stage1(x)
        x = self.stage2(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with skip connections
        identity = x
        x = self.fc1(x)
        x = x + self.fc1_skip(identity)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.output(x)
        return x
    




# ================================= DATA LOADER ===================================
class EyeAngleDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, augment=None, scale_output=None):
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
        self.augment = augment
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
        
        if self.augment:
            image, labels = self.augment(image, labels)
        if self.transform:
            image = self.transform(image)
        if self.scale_output is not None:
            labels = ((labels - self.scale_output[0]) / self.scale_output[1]).astype(np.float32)
            # labels *= self.scale_output
                   
        return image, labels
    

# augmentation utilities
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Generate noise and add it to the tensor
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
    

class RotateScaleTransform:
    def __init__(self, angle_range=(-30, 30), scale_range=(0.8, 1.2)):
        self.angle_range = angle_range
        self.scale_range = scale_range

    def __call__(self, image, labels):
        # Convert keypoints to numpy array if needed
        labels = np.array(labels)

        # 1. Randomly sample rotation angle and scale factor
        rotation_angle = np.random.uniform(*self.angle_range)
        scale_factor = np.random.uniform(*self.scale_range)

        # 2. Apply transformations to the image
        # Rotate and scale the image
        width, height = image.size
        image = image.rotate(rotation_angle, resample=Image.BILINEAR)
        image = image.resize((int(width * scale_factor), int(height * scale_factor)), resample=Image.BILINEAR)

        # 3. Adjust keypoints based on rotation and scaling
        # Rotation matrix
        rotation_rad = math.radians(rotation_angle)
        cos_theta = math.cos(rotation_rad)
        sin_theta = -math.sin(rotation_rad)

        # Center of the image (for rotation)
        cx, cy = width / 2, height / 2

        # Apply rotation and scaling to each keypoint
        new_keypoints = []
        keypoints = [(x, y) for x, y in zip(labels[0:6:2], labels[1:6:2])]
        for (x, y) in keypoints:
            # Translate to origin for rotation
            x -= cx
            y -= cy

            # Rotate point
            x_new = x * cos_theta - y * sin_theta
            y_new = x * sin_theta + y * cos_theta

            # Scale point
            x_new *= scale_factor
            y_new *= scale_factor

            # Translate back to original center and add to list
            new_keypoints.append((x_new + cx, y_new + cy))
        new_keypoints = np.array(new_keypoints).flatten()

        # 4. Update the angle (if relevant)
        # Add rotation angle to original angle
        new_heading = labels[-1] - rotation_rad
        new_labels = new_keypoints.tolist() + labels[6:8].tolist() + [new_heading]

        return image, np.array(new_labels).astype(np.float32)


# =============================== Utilities ===================================
def get_mu_sigma_images(dir, transform):
    imgs = os.listdir(dir)
    img = transform(Image.open(os.path.join(dir, imgs[0])).convert("L"))
    all_img = np.zeros((len(imgs), img.shape[0], img.shape[1]))
    for i, img in enumerate(imgs):
        all_img[i, :, :] = transform(Image.open(os.path.join(dir, img)).convert("L")).mean()
    mu = all_img.mean()
    sigma = all_img.std()
    return mu, sigma

def get_mu_sigma_outputs(dir, keys=None):
    if keys is None:
        keys = [
            "left_eye_x_position",
            "left_eye_y_position",
            "right_eye_x_position",
            "right_eye_y_position",
            "yolk_x_position",
            "yolk_y_position",
            "left_eye_angle",
            "right_eye_angle",
            "heading_angle"
            ]
        
    ff = os.listdir(dir)
    for i, fname in enumerate(ff):
        with open(os.path.join(dir, fname), "r") as f:
            _annotations = json.load(f)[0]
            if i == 0:
                all_output = np.zeros((len(ff), len(keys)))
            for j, k in enumerate(keys):
                all_output[i, j] = _annotations[k]
    mu = np.mean(all_output, axis=0)
    sigma = np.std(all_output, axis=0)
    return mu, sigma