## Copy a random subset of the training data into the val folder
from settings import IMG_DIR, ANNOT_DIR, IMG_VAL_DIR, ANNOT_VAL_DIR
import os
import numpy as np

imgs = os.listdir(IMG_DIR)
ridx = np.random.choice(np.arange(0, len(imgs)), 50, replace=False)

for idx in ridx:
    img_file = imgs[idx]

    img_path = os.path.join(IMG_DIR, img_file)
    label_path = os.path.join(ANNOT_DIR, img_file.replace(".png", ".json"))
    new_img_path = os.path.join(IMG_VAL_DIR, img_file)
    new_lab_path = os.path.join(ANNOT_VAL_DIR, img_file.replace(".png", ".json"))

    os.system(f"mv {img_path} {new_img_path}")
    os.system(f"mv {label_path} {new_lab_path}")