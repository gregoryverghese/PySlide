import os
import glob

import cv2
import numpy as np



def calculate_std_mean(patch_files,patch_path):

    if patch_path is not None:
        patches = glob.glob(os.path.join(path,'*'))
    shape = cv2.imread(images[0]).shape
    channels = shape[-1]
    chnl_values = np.zeros((channels))
    chnl_values_sqrt = np.zeros((chnl_values))
    pixel_nums = len(patches)*shape[0]*shape[1]
    print('total number pixels: {}'.format(pixel_nums))

    for path in images:
        patch = cv2.imread(path)
        patch = (patch/255.0).astype('float64')
        chnl_values += np.sum(patch, axis=(0,1), dtype='float64')
    mean=chnl_values/pixel_nums

    for path in images:
        patch = cv2.imread(path)
        patch = (patch/255.0).astype('float64')
        chnl_values_sqrt += np.sum(np.square(image-mean), axis=(0,1), dtype='float64')
    std=np.sqrt(chnl_values_sqrt/pixel_nums, dtype='float64')
    
    print('mean: {}, std: {}'.format(mean, std))
    return mean, std 



def calculate_weights(mask_files,num_cls,mask_path=None):

    if mask_path is not None:
        mask_files = glob.glob(os.path.join(maskPath,'*'))
    total = {c:0 for c in range(numClasses)}
    for f in mask_files:
        mask = cv2.imread(f)
        pixels = mask.reshape(-1)
        classes = np.unique(pixels, return_counts=True)
        pixelDict = dict(list(zip(*classes)))     
        for k, v in pixelDict.items():
            total[k] = total[k] + v 
    if numClasses==2:
        weight = total[0]/total[1]
    else:
        weight = [1/v for v in list(total.values())]
    return weight
    






























