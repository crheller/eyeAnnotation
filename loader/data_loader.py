import sys
sys.path.append("/home/charlie/code/eyeAnnotation")
from settings import IMG_DIR
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

    pyimg, frame_idx, n_frames = reader.read_random(rolipath)
        
    # zoom and make ego centric
    imgpath = imh.save_egocentric(pyimg, ds, frame_idx, n_frames)

    return imgpath