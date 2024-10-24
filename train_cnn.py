import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import json
from CNN import CustomCNN, EyeAngleDataset, DeepCNN
from settings import IMG_DIR, ANNOT_DIR
import matplotlib.pyplot as plt
import math

def train_model(model, train_loader, num_epochs=50, device="cuda"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    model = model.to(device)
    # best_val_loss = float('inf')
    save_loss = np.zeros(num_epochs)
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
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print('--------------------')
        # scheduler.step(avg_train_loss)
    return save_loss

# build model, define transform, call training function
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Create datasets
train_dataset = EyeAngleDataset(
    image_dir=IMG_DIR,
    label_dir=ANNOT_DIR,
    transform=transform,
    scale_output=np.array([1, 1, 1, 1, 1, 1, 100, 100, 50])
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

# Initialize model
# model = CustomCNN(dropout_rate=0.5)
model = DeepCNN(dropout_rate=0.2)

# Train model
loss = train_model(model, train_loader, num_epochs=500, device="cuda:1")

f, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.plot(loss)

# save the trained model, load it, run inference
# Specify a path
PATH = "SavedModels/DeepCNN.pt"

# Save
torch.save(model.state_dict(), PATH)

# Load
device = "cuda:0"
loadedmodel = DeepCNN()
loadedmodel.load_state_dict(torch.load(PATH, weights_only=True))
loadedmodel.eval()
loadedmodel.to(device)

ii = iter(train_loader)
input, output = next(ii)
img = input[[6]].to(device)

with torch.no_grad():
  pred = loadedmodel(img)
pred = np.array(pred.to("cpu"))
imgcpu = np.array(img.to("cpu"))[0, 0, :, :]


def create_vector(x_start, y_start, angle_rad, length):
    
    # Calculate the change in x and y using trigonometry
    x_end = x_start + length * math.cos(angle_rad)
    y_end = y_start + length * math.sin(angle_rad)
    
    return (x_end, y_end)

lx, ly = pred[0, 0], pred[0, 1]
rx, ry = pred[0, 2], pred[0, 3]
yx, yy = pred[0, 4], pred[0, 5]
left_eye_angle = pred[0, 6] / 100
right_eye_angle = pred[0, 7] / 100
heading_angle = pred[0, 8] / 50

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