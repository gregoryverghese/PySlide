import cv2
import numpy as np
from skimage.morphology import disk
from skimage.filters.rank import entropy


def image_entropy(patch):
    gray=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY)
    entr=entropy(gray,disk(10))
    avg_entr=np.mean(entr)
    return avg_entr


def remove_black(patch,
                 threshold=110,
                 max_value=255,
                 area_thresh=0.2):
    gray=cv2.cvtColor(image,cv2.COLOR_RGBGRAY)
    _,thresh=cv2.threshold(image,threshold,max_value,cv2.THRESH_BINARY)
    _,counts=np.unique(thresh,return_counts=True)
    area_proportion=counts[0]/counts[0]+counts[1]
    return area_proportion
