import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import json
from CNN import SmallNetTransformPipeline, EyeAngleDatasetKeyPoints, ShallowCNN, DeepCNN, get_mu_sigma_images, get_mu_sigma_outputs
from settings import IMG_DIR, ANNOT_DIR, IMG_VAL_DIR, ANNOT_VAL_DIR, IMG_X_DIM, IMG_Y_DIM
import matplotlib.pyplot as plt
import math

def train_model(model, train_loader, val_loader, l2_penalty=None, num_epochs=50, device="cuda"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=l2_penalty)

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
        # scheduler.step(avg_train_loss)

    return save_loss, save_val_loss


# load training inputs / outputs and figure out how to best standardize
tensor_transform = transforms.Compose([
    transforms.ToTensor()
])
u_input, sd_input = get_mu_sigma_images(IMG_DIR, transform=tensor_transform)
# scale_ouput = get_mu_sigma_outputs(ANNOT_DIR)
scale_output = (0.5, 0.1)
# build model, define transform, call training function
transform = SmallNetTransformPipeline(u_input, sd_input, augment=True)
val_transform = SmallNetTransformPipeline(u_input, sd_input, augment=False)

# Create datasets
train_dataset = EyeAngleDatasetKeyPoints(
    image_dir=IMG_DIR,
    label_dir=ANNOT_DIR,
    transform=transform,
    # scale_output=scale_output
)
validation_dataset = EyeAngleDatasetKeyPoints(
    image_dir=IMG_VAL_DIR,
    label_dir=ANNOT_VAL_DIR,
    transform=val_transform,
    # scale_output=scale_output
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True, num_workers=4)

# Initialize model
model, PATH = ShallowCNN(dropout_rate=0.1, num_outputs=24), "SavedModels/ShallowNetKeyPoints.pt" # good parity between training / test, but doesn't get too low (and eventually overfits too)
# model, PATH = DeepCNN(dropout_rate=0.3, num_outputs=30), "SavedModels/DeepNetKeyPoints.pt" # can get training loss lower, but seems to overfit slightly more

# Train model
l2_penalty = 0 #1e-5
n_epochs = 1000
loss, val_loss = train_model(model, train_loader, val_loader, l2_penalty=l2_penalty, num_epochs=n_epochs, device="cuda:1")

f, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.plot(val_loss, lw=0.5, label="test")
ax.plot(loss, lw=0.5, label="train")
ax.legend()
ax.set_xlabel("epoch")
# ax.set_ylim((None, 1))

# Save
torch.save(model.state_dict(), PATH)

# Load
device = "cuda:0"
# loadedmodel = DeepCNN()
loadedmodel = ShallowCNN(num_outputs=24)
loadedmodel.load_state_dict(torch.load(PATH, weights_only=True))
loadedmodel.eval()
loadedmodel.to(device)

ii = iter(val_loader)
# ii = iter(train_loader)
input, output = next(ii)

for i in range(0, 16):
    img = input[[i]].to(device)

    with torch.no_grad():
        pred = loadedmodel(img)
        # pred = output[[i], :]
    pred = np.array(pred.to("cpu"))
    # pred = (pred * scale_output[1]) + scale_output[0]
    imgcpu = np.array(img.to("cpu"))[0, 0, :, :]

    # xsize, ysize = imgcpu.shape
    left_keypoints = pred[0, :6] * 60
    right_keypoints = pred[0, 6:12] * 60
    heading_keypoints = pred[0, 12:] * 60

    f, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.imshow(imgcpu)
    for i in np.arange(0, len(left_keypoints), step=2):
        ax.scatter(left_keypoints[i], left_keypoints[i+1], s=25, c="cyan")
    for i in np.arange(0, len(right_keypoints), step=2):
        ax.scatter(right_keypoints[i], right_keypoints[i+1], s=25, c="tab:orange")
    for i in np.arange(0, len(heading_keypoints), step=2):
        ax.scatter(heading_keypoints[i], heading_keypoints[i+1], s=25, c="red")
    # ax.set_title(f"h: {hed}, l: {led}, r: {red}")