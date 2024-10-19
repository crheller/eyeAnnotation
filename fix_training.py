"""
One off script to fix annotations, if needed
"""
import json
from settings import IMG_DIR
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math


# test load an image to see if the annotations look correct
fname = "20240802_100046_1_41708"
img = np.array(Image.open(os.path.join(IMG_DIR, "images", f"{fname}.png")))
with open(os.path.join(IMG_DIR, f"{fname}.json"), "r") as f:
    annot = json.load(f)[0]


def create_vector(x_start, y_start, angle_rad, length):
    
    # Calculate the change in x and y using trigonometry
    x_end = x_start + length * math.cos(angle_rad)
    y_end = y_start + length * math.sin(angle_rad)
    
    return (x_end, y_end)

lx, ly = annot["left_eye_x_position"], annot["left_eye_y_position"]
rx, ry = annot["right_eye_x_position"], annot["right_eye_y_position"]
yx, yy = annot["yolk_x_position"], annot["yolk_y_position"]
lex, ley = create_vector(lx, ly, annot["left_eye_angle"]+annot["heading_angle"], 5)
rex, rey = create_vector(rx, ry, annot["right_eye_angle"]+annot["heading_angle"], 5)
yex, yey = create_vector(yx, yy, annot["heading_angle"], 15)

f, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.imshow(img)
ax.scatter(lx, ly, c="tab:orange")
ax.plot([lx, lex], [ly, ley], "tab:orange")
ax.scatter(rx, ry, c="tab:blue")
ax.plot([rx, rex], [ry, rey], "tab:blue")
ax.scatter(yx, yy, c="r")
ax.plot([yx, yex], [yy, yey], "r")
