import os
import glob

import cv2
import numpy as np



def calculate_std_mean(patch_path):

    if patch_path is not None:
        patches = glob.glob(os.path.join(patch_path,'*'))
    shape = cv2.imread(patches[0]).shape
    channels = shape[-1]
    chnl_values = np.zeros((channels))
    print(chnl_values)
    chnl_values_sqrt = np.zeros((channels))
    pixel_nums = len(patches)*shape[0]*shape[1]
    print('total number pixels: {}'.format(pixel_nums))

    for path in patches:
        patch = cv2.imread(path)
        patch = (patch/255.0).astype('float64')
        chnl_values += np.sum(patch, axis=(0,1), dtype='float64')
    mean=chnl_values/pixel_nums

    for path in patches:
        patch = cv2.imread(path)
        patch = (patch/255.0).astype('float64')
        chnl_values_sqrt += np.sum(np.square(patch-mean), axis=(0,1), dtype='float64')
    std=np.sqrt(chnl_values_sqrt/pixel_nums, dtype='float64')
    
    print('mean: {}, std: {}'.format(mean, std))
    return mean, std 



def calculate_weights(mask_path,num_cls):

    if mask_path is not None:
        mask_files = glob.glob(os.path.join(mask_path,'*'))
    cls_nums = {c:0 for c in range(num_cls)}
    for f in mask_files:
        mask = cv2.imread(f)
        pixels = mask.reshape(-1)
        classes = np.unique(pixels, return_counts=True)
        pixelDict = dict(list(zip(*classes)))     
        for k, v in pixelDict.items():
            cls_nums[k] = cls_nums[k] + v
    total = sum(list(cls_nums.values()))
    weights = [v/total for v in list(cls_nums.values())]
    weights = [1/w for w in weights]
    return weights
    






























