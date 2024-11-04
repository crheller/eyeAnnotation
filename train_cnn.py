import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import json
from CNN import ShallowCNN, EyeAngleDataset, DeepCNN, CustomCNN, AddGaussianNoise, RotateScaleTransform, get_mu_sigma_images, get_mu_sigma_outputs
from settings import IMG_DIR, ANNOT_DIR, IMG_VAL_DIR, ANNOT_VAL_DIR, IMG_X_DIM, IMG_Y_DIM
import matplotlib.pyplot as plt
import math

def train_model(model, train_loader, val_loader, l2_penalty=None, num_epochs=50, device="cuda"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_penalty)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    model = model.to(device)
    # best_val_loss = float('inf')
    save_loss = np.zeros(num_epochs)
    save_val_loss = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        save_loss[epoch] = avg_train_loss

        # add a validation loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient computation
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()  # Accumulate validation loss
        val_loss /= len(val_loader)
        save_val_loss[epoch] = val_loss

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print('--------------------')
        print(f'Val loss: {val_loss}')
        # scheduler.step(avg_train_loss)
    return save_loss, save_val_loss


# load training inputs / outputs and figure out how to best standardize
tensor_transform = transforms.Compose([
    transforms.ToTensor()
])
u_input, sd_input = get_mu_sigma_images(IMG_DIR, transform=tensor_transform)
scale_output = get_mu_sigma_outputs(ANNOT_DIR)

# build model, define transform, call training function
transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),
    # transforms.Resize(256),         # Resize the shortest side to 256 pixels
    # transforms.CenterCrop(224),     # Then crop the center 224x224 portion
    transforms.ToTensor(),
    transforms.Resize((IMG_X_DIM, IMG_Y_DIM)),
    transforms.Normalize(mean=[u_input], std=[sd_input]),
    # AddGaussianNoise(mean=0.0, std=0.1)
])
augment = RotateScaleTransform(angle_range=(-90, 90), scale_range=(0.7, 1.2))
val_transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),
    # transforms.Resize(256),         # Resize the shortest side to 256 pixels
    # transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[u_input], std=[sd_input]),
])

# Create datasets
train_dataset = EyeAngleDataset(
    image_dir=IMG_DIR,
    label_dir=ANNOT_DIR,
    transform=transform,
    augment=augment,
    scale_output=scale_output
)
validation_dataset = EyeAngleDataset(
    image_dir=IMG_VAL_DIR,
    label_dir=ANNOT_VAL_DIR,
    transform=val_transform,
    scale_output=scale_output
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(validation_dataset, batch_size=16, shuffle=True, num_workers=4)

# Initialize model
# model = CustomCNN(dropout_rate=0.3)
# model = DeepCNN(dropout_rate=0.3)
model = ShallowCNN(dropout_rate=0.1)
# model = ResNet50Regression(freeze_backbone=True)

# Train model
l2_penalty = 1e-5
n_epochs = 1000
loss, val_loss = train_model(model, train_loader, val_loader, l2_penalty=l2_penalty, num_epochs=n_epochs, device="cuda:1")

f, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.plot(val_loss)
ax.plot(loss)
# ax.set_ylim((None, 1))

# save the trained model, load it, run inference
# Specify a path
PATH = "SavedModels/ShallowCNN.pt"

# Save
torch.save(model.state_dict(), PATH)

# Load
device = "cuda:0"
loadedmodel = ShallowCNN()
loadedmodel.load_state_dict(torch.load(PATH, weights_only=True))
loadedmodel.eval()
loadedmodel.to(device)

def create_vector(x_start, y_start, angle_rad, length):
    
    # Calculate the change in x and y using trigonometry
    x_end = x_start + length * math.cos(angle_rad)
    y_end = y_start + length * math.sin(angle_rad)
    
    return (x_end, y_end)

def invert_scaling(pred, scaler):
    pred = (pred * scaler[1]) + scaler[0]
    return pred


ii = iter(val_loader)
# ii = iter(train_loader)
input, output = next(ii)

for i in range(1, 10):
    img = input[[i]].to(device)

    with torch.no_grad():
        pred = loadedmodel(img)
        # pred = output[[i], :]
    pred = np.array(pred.to("cpu"))
    imgcpu = np.array(img.to("cpu"))[0, 0, :, :]

    pred = invert_scaling(pred, scale_output)

    lx, ly = pred[0, 0], pred[0, 1]
    rx, ry = pred[0, 2], pred[0, 3]
    yx, yy = pred[0, 4], pred[0, 5]
    left_eye_angle = pred[0, 6] #/ 100
    right_eye_angle = pred[0, 7] #/ 100
    heading_angle = pred[0, 8] #/ 50

    lex, ley = create_vector(lx, ly, left_eye_angle + heading_angle, 5)
    rex, rey = create_vector(rx, ry, right_eye_angle + heading_angle, 5)
    yex, yey = create_vector(yx, yy, heading_angle, 15)

    f, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.imshow(imgcpu)
    ax.scatter(lx, ly, c="tab:orange")
    ax.plot([lx, lex], [ly, ley], "tab:orange")
    ax.scatter(rx, ry, c="tab:blue")
    ax.plot([rx, rex], [ry, rey], "tab:blue")
    ax.scatter(yx, yy, c="r")
    ax.plot([yx, yex], [yy, yey], "r")
    hed = np.round(np.rad2deg(heading_angle), 2)
    led = np.round(np.rad2deg(left_eye_angle), 2)
    red = np.round(np.rad2deg(right_eye_angle), 2)
    ax.set_title(f"h: {hed}, l: {led}, r: {red}")