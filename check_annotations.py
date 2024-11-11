"""
Load annotations and make sure they seem correct.
"""

import json
from settings import IMG_DIR, ANNOT_DIR
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

# IMG_DIR = "/data/charlie/data/EyeAnnotations/rotated_images"
# ANNOT_DIR = "/data/charlie/data/EyeAnnotations/rotated_annotations"
# test load an image to see if the annotations look correct
def create_vector(x_start, y_start, angle_rad, length):
    
    # Calculate the change in x and y using trigonometry
    x_end = x_start + length * math.cos(angle_rad)
    y_end = y_start + length * math.sin(angle_rad)
    
    return (x_end, y_end)

def create_keypoints(x_mid, y_mid, angle_rad, length, nkeypoints):

    length_step = length / nkeypoints
    steps = np.arange(0, length+length_step, length_step)

    p_keypoints = []
    n_keypoints = []
    for s in steps:
        x = x_mid + s * math.cos(angle_rad)
        y = y_mid + s * math.sin(angle_rad)
        p_keypoints.append((x, y))

        if s != 0:
            x = x_mid - s * math.cos(angle_rad)
            y = y_mid - s * math.sin(angle_rad)
            n_keypoints.append((x, y))
    
    return n_keypoints[::-1] + p_keypoints

def is_point_right_of_vector(A, B, P):
    """
    Check if point P is to the right of the vector AB.
    
    Parameters:
    A: tuple - coordinates of point A (x_a, y_a)
    B: tuple - coordinates of point B (x_b, y_b)
    P: tuple - coordinates of point P (x_p, y_p)
    
    Returns:
    True if point P is to the right of vector AB, False otherwise.
    """
    x_a, y_a = A
    x_b, y_b = B
    x_p, y_p = P
    
    # Calculate the cross product of AB and AP
    cross_product = (x_b - x_a) * (y_p - y_a) - (y_b - y_a) * (x_p - x_a)
    
    # If the cross product is negative, the point is to the right
    return cross_product > 0

for fname in os.listdir(IMG_DIR)[30:40]:
    fname = fname.strip(".png")
    img = np.array(Image.open(os.path.join(IMG_DIR, f"{fname}.png")))
    with open(os.path.join(ANNOT_DIR, f"{fname}.json"), "r") as f:
        annot = json.load(f)[0]


    lx, ly = annot["left_eye_x_position"], annot["left_eye_y_position"]
    rx, ry = annot["right_eye_x_position"], annot["right_eye_y_position"]
    yx, yy = annot["yolk_x_position"], annot["yolk_y_position"]

    lex, ley = create_vector(lx, ly, annot["left_eye_angle"]+annot["heading_angle"], 5)
    rex, rey = create_vector(rx, ry, annot["right_eye_angle"]+annot["heading_angle"], 5)
    yex, yey = create_vector(yx, yy, annot["heading_angle"], 15)

    left_keypoints = create_keypoints(lx, ly, annot["left_eye_angle"]+annot["heading_angle"], 5, 2)
    right_keypoints = create_keypoints(rx, ry, annot["right_eye_angle"]+annot["heading_angle"], 5, 2)
    heading_keypoints = create_keypoints(yx, yy, annot["heading_angle"], 15, 3)

    right_is_right = is_point_right_of_vector((yx, yy), (yex, yey), (lx, ly))

    if right_is_right:
        iscorrect = "correct"
    else:
        iscorrect = "left is right"

    f, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.imshow(img)
    # ax.scatter(lx, ly, c="tab:orange")
    # ax.plot([lx, lex], [ly, ley], "tab:orange")
    # ax.scatter(rx, ry, c="tab:blue")
    # ax.plot([rx, rex], [ry, rey], "tab:blue")
    # ax.scatter(yx, yy, c="r")
    # ax.plot([yx, yex], [yy, yey], "r")
    for kk in left_keypoints:
        ax.scatter(kk[0], kk[1], s=25)
    for kk in right_keypoints:
        ax.scatter(kk[0], kk[1], s=25)
    for kk in heading_keypoints:
        ax.scatter(kk[0], kk[1], s=25)
    hed = round(np.rad2deg(annot['heading_angle']), 2)
    led = round(np.rad2deg(annot['left_eye_angle']), 2)
    red = round(np.rad2deg(annot['right_eye_angle']), 2)
    ax.set_title(f"h: {hed}, l: {led}, r: {red}, {iscorrect}")

    # plt.close(f)


# check that all right eye angles are actually to the right
# also plot distribution of eye / heading angles
labeled = os.listdir(IMG_DIR)
left_angles = np.zeros(len(labeled))
right_angles = np.zeros(len(labeled))
heading_angles = np.zeros(len(labeled))
correct_assignment = np.zeros(len(labeled))
for i, fname in enumerate(labeled):
    fname = fname.strip(".png")
    img = np.array(Image.open(os.path.join(IMG_DIR, f"{fname}.png")))
    with open(os.path.join(ANNOT_DIR, f"{fname}.json"), "r") as f:
        annot = json.load(f)[0]


    lx, ly = annot["left_eye_x_position"], annot["left_eye_y_position"]
    rx, ry = annot["right_eye_x_position"], annot["right_eye_y_position"]
    yx, yy = annot["yolk_x_position"], annot["yolk_y_position"]

    yex, yey = create_vector(yx, yy, annot["heading_angle"], 15)

    right_is_right = is_point_right_of_vector((yx, yy), (yex, yey), (rx, ry))

    correct_assignment[i] = right_is_right
    left_angles[i] = annot["left_eye_angle"]
    right_angles[i] = annot["right_eye_angle"]
    heading_angles[i] = annot["heading_angle"]

print(f"{int(np.sum(correct_assignment))}/{len(labeled)} labeled correctly")

f, ax = plt.subplots(1, 1, figsize=(6, 3))

ax.hist(heading_angles, bins=np.arange(-np.pi, np.pi, step=0.1), histtype="step", lw=2, label="heading")
ax.hist(left_angles, bins=np.arange(-np.pi, np.pi, step=0.1), histtype="step", lw=2, label="left_eye")
ax.hist(right_angles, bins=np.arange(-np.pi, np.pi, step=0.1), histtype="step", lw=2, label="right_eye")
ax.legend(frameon=False)
ax.set_xlabel("Angle (rads)")

f, ax = plt.subplots(1, 1, figsize=(6, 3))

ax.hist(left_angles-right_angles, bins=np.arange(-np.pi/2, np.pi/2, step=0.02), histtype="step", lw=2, label="left minus right")
ax.legend(frameon=False)
ax.set_xlabel("Angle (rads)")
