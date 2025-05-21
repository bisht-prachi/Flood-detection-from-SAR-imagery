import os
import numpy as np

def get_filename(filepath):
    """
    Extracts the filename from a given path.
    """
    return os.path.split(filepath)[1]

def to_rgb(vv_image, vh_image):
    """
    Creates an RGB image using VV and VH polarizations.
    The third channel is 1 - (VV/VH ratio).
    """
    ratio_image = np.clip(np.nan_to_num(vv_image / vh_image, 0), 0, 1)
    rgb_image = np.stack((vv_image, vh_image, 1 - ratio_image), axis=2)
    return rgb_image
