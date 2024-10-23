import sys
sys.path.append("/home/charlie/code/eyeAnnotation")
import numpy as np
import h5py
from scipy.ndimage import rotate
import paths
from settings import IMG_TMP, IMG_X_DIM, IMG_Y_DIM
import os
import json
import matplotlib.pyplot as plt
import datetime

def make_egocentric_fast(pyimg, xpos, ypos, heading=None, rotate=False):
    if rotate:
        rdeg = np.rad2deg(heading)
        _img = pyimg[ypos-50:ypos+50, xpos-50:xpos+50]
        newimg = rotate(_img, rdeg)
        mx, my = (int(t/2) for t in newimg.shape)
        mmx = int(IMG_X_DIM / 2)
        mmy = int(IMG_Y_DIM / 2)
        final_img = newimg[mx-mmx:mx+mmx, my-mmy:my+mmy]
    else:
        mmx = int(IMG_X_DIM / 2)
        mmy = int(IMG_Y_DIM / 2)
        final_img = pyimg[ypos-mmy:ypos+mmy, xpos-mmx:xpos+mmx]

    return final_img

def make_egocentric(pyimg, ds, frame_idx, n_frames, rotate=False):
    bf = h5py.File(paths.locate(ds[0], ds[1], f"behavior_{ds[2]}.h5", raw=True))
    # save divisor
    save_divisor = int(len(bf["heading"]) / n_frames)
    camera_frame_index = bf["camera_frame_index"][1:500] # to speed up, just read the first 500. Just need to figure out offset

    save_idx = np.argwhere(np.mod(camera_frame_index, save_divisor)==0).squeeze()[0]
    save_idx = np.arange(save_idx, n_frames*save_divisor, save_divisor).astype(int)
    heading = bf["heading"][save_idx[frame_idx]]
    xpos = bf["fish_anchor_x"][save_idx[frame_idx]] - bf["offset_x"][save_idx[frame_idx]]
    ypos = bf["fish_anchor_y"][save_idx[frame_idx]] - bf["offset_y"][save_idx[frame_idx]]

    if rotate:
        rdeg = np.rad2deg(heading)
        _img = pyimg[ypos-50:ypos+50, xpos-50:xpos+50]
        newimg = rotate(_img, rdeg)
        mx, my = (int(t/2) for t in newimg.shape)
        bf.close()
        mmx = int(IMG_X_DIM / 2)
        mmy = int(IMG_Y_DIM / 2)
        final_img = newimg[mx-mmx:mx+mmx, my-mmy:my+mmy]
    else:
        mmx = int(IMG_X_DIM / 2)
        mmy = int(IMG_Y_DIM / 2)
        final_img = pyimg[ypos-mmy:ypos+mmy, xpos-mmx:xpos+mmx]

    return final_img

def save_egocentric(pyimg, ds, frame_idx, n_frames, rotate=False):

    final_img = make_egocentric(pyimg, ds, frame_idx, n_frames, rotate=False)

    if final_img.size==0:
        return False

    else:
        # save the image into a temp directory (gets moved up when it's annotated)
        savepath = os.path.join(IMG_TMP, f"{ds[0]}_{ds[-1]}_{frame_idx}.png")
        plt.imsave(savepath, final_img, cmap="gray")
        return savepath