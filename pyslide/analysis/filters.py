import cv2
import numpy as np
from skimage.morphology import disk
from skimage.filters.rank import entropy


def image_entropy(patch):
    gray=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY)
    entr=entropy(gray,disk(10))
    avg_entr=np.mean(entr)
    return avg_entr
