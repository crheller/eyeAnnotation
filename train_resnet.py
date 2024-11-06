import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import json
from CNN import ResNetTransformPipeline, EyeAngleDatasetResNet, ResNet50Regression, get_mu_sigma_images, get_mu_sigma_outputs
from settings import IMG_DIR, ANNOT_DIR, IMG_VAL_DIR, ANNOT_VAL_DIR, IMG_X_DIM, IMG_Y_DIM
import matplotlib.pyplot as plt
import math

def train_model(model, train_loader, val_loader, l2_penalty=None, num_epochs=50, device="cuda"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=l2_penalty)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    model = model.to(device)
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
        scheduler.step(avg_train_loss)

    return save_loss, save_val_loss


# load training inputs / outputs and figure out how to best standardize
tensor_transform = transforms.Compose([
    transforms.ToTensor()
])
u_input, sd_input = get_mu_sigma_images(IMG_DIR, transform=tensor_transform)
scale_output = get_mu_sigma_outputs(ANNOT_DIR)

# build model, define transform, call training function
transform = ResNetTransformPipeline(u_input, sd_input, augment=True)
val_transform = ResNetTransformPipeline(u_input, sd_input, augment=False)

# Create datasets
train_dataset = EyeAngleDatasetResNet(
    image_dir=IMG_DIR,
    label_dir=ANNOT_DIR,
    transform=transform,
    scale_output=None
)
validation_dataset = EyeAngleDatasetResNet(
    image_dir=IMG_VAL_DIR,
    label_dir=ANNOT_VAL_DIR,
    transform=val_transform,
    scale_output=None
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True, num_workers=4)

# Initialize model
model = ResNet50Regression(freeze_backbone=False, dropout_rate=0.0)

# Train model
l2_penalty = 1e-4
n_epochs = 100
loss, val_loss = train_model(model, train_loader, val_loader, l2_penalty=l2_penalty, num_epochs=n_epochs, device="cuda:1")

f, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.plot(val_loss, lw=0.5, label="test")
ax.plot(loss, lw=0.5, label="train")
ax.legend()
ax.set_xlabel("epoch")
# ax.set_ylim((None, 1))

# save the trained model, load it, run inference
# Specify a path
PATH = "SavedModels/ResNet.pt"

# Save
torch.save(model.state_dict(), PATH)

# Load
device = "cuda:0"
loadedmodel = ResNet50Regression()
loadedmodel.load_state_dict(torch.load(PATH, weights_only=True))
loadedmodel.eval()
loadedmodel.to(device)

def create_vector(x_start, y_start, angle_rad, length):
    
    # Calculate the change in x and y using trigonometry
    x_end = x_start + length * math.cos(angle_rad)
    y_end = y_start + length * math.sin(angle_rad)
    
    return (x_end, y_end)

ii = iter(val_loader)
# ii = iter(train_loader)
input, output = next(ii)

for i in range(0, 16):
    img = input[[i]].to(device)

    with torch.no_grad():
        pred = loadedmodel(img)
        # pred = output[[i], :]
    pred = np.array(pred.to("cpu"))
    imgcpu = np.array(img.to("cpu"))[0, 0, :, :]

    # xsize, ysize = imgcpu.shape
    pred[0, :6] = pred[0, :6] * 224
    lx, ly = pred[0, 0], pred[0, 1]
    rx, ry = pred[0, 2], pred[0, 3]
    yx, yy = pred[0, 4], pred[0, 5]
    left_eye_angle = pred[0, 6] #/ 100
    right_eye_angle = pred[0, 7] #/ 100
    heading_angle = pred[0, 8] #/ 50

    lex, ley = create_vector(lx, ly, left_eye_angle + heading_angle, 15)
    rex, rey = create_vector(rx, ry, right_eye_angle + heading_angle, 15)
    yex, yey = create_vector(yx, yy, heading_angle, 25)

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