import cv2
import numpy as np
from skimage.morphology import disk
from skimage.filters.rank import entropy


def image_entropy(patch):
    gray=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY)
    entr=entropy(gray,disk(10))
    avg_entr=np.mean(entr)
    return avg_entr


def entropy(tile, threshold):
    avg_entropy=image_entropy(tile)
    if avg_entropy<threshold:
        return True
    

def tile_intensity(tile, threshold, channel=None):
    print(tile)
    print('grr',np.mean(tile))
    if channel is not None:
        if np.mean(tile[:,:,channel]) > threshold:
            return True
     
    elif channel is None:
        if np.mean(tile)>threshold:
            return True


def remove_black(patch,
                 threshold=60,
                 max_value=255,
                 area_thresh=0.2):
    n=len(patch._patches)
    print('n',n)
    patches=patch._patches.copy()
    for image,p in patch.extract_patches():
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        _,thresh=cv2.threshold(gray,threshold,max_value,cv2.THRESH_BINARY)
        values,counts=np.unique(thresh,return_counts=True)
        if len(values)==2:
            area_proportion=counts[0]/(counts[0]+counts[1])
            if area_proportion>area_thresh:
                patches.remove(p)

        elif len(values)==1 and values[0]==0:
            patches.remove(p)
        elif len(values)==1 and values[0]==255:
            continue

    patch._patches=patches
    n_removed=n-len(patch._patches)
    print(f'Black: N patches removed:{n_removed}')
    return patch


def remove_blue(patch,
                area_thresh=0.2,
                lower_blue=[100,150,0],
                upper_blue=[130,255,255]
                ):

    n=len(patch._patches)
    patches=patch._patches.copy()
    for image,p in patch.extract_patches():
        hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        mask=cv2.inRange(hsv,np.array(lower_blue),np.array(upper_blue))
        values,counts=np.unique(mask,return_counts=True)

        if len(values)==2:
            area_proportion=counts[1]/(counts[0]+counts[1])
            if area_proportion>area_thresh:
                patches.remove(p)

        elif len(values)==1 and values[0]==255:
            patches.remove(p)

        elif len(values)==1 and values[0]==0:
            continue

    patch._patches=patches
    n_removed=n-len(patch._patches)
    print(f'Blue: N patches removed:{n_removed}')
    return patch

