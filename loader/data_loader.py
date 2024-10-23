import sys
sys.path.append("/home/charlie/code/eyeAnnotation")
import pymongo as pym
import paths
import numpy as np
from datasets import datasets
from scipy.ndimage import rotate
import imgProcessing.img_helpers as imh
import imgProcessing.reader as reader

def load_new_image():
    """
    Load a new image. Ensure that it is not already labeled (check database entries)
    """

    # select a dataset
    ds = datasets[np.random.choice(range(len(datasets)))]
    rolipath = paths.locate(ds[0], ds[1], f"ir_{ds[2]}.roli", raw=True)
       
    # zoom and make ego centric
    # read frames until we get one that is not empty (can happen if tracking was lost)
    imgpath = False
    while imgpath == False:
        pyimg, frame_idx, n_frames = reader.read_random(rolipath)
        imgpath = imh.save_egocentric(pyimg, ds, frame_idx, n_frames)

    return imgpath