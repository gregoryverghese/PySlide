'''
utilities.py: useful functions
'''

import numpy as np
import xml.etree.ElementTree as ET
import seaborn as sns
from itertools import chain


def mask2rgb(mask):
    n_classes=len(np.unique(mask))
    colors=sns.color_palette('hls',n_classes)
    rgb_mask=np.zeros(mask.shape+(3,))
    for c in range(1,n_classes+1):
        t=(mask==c)
        rgb_mask[:,:,0][t]=colors[c-1][0]
        rgb_mask[:,:,1][t]=colors[c-1][1]
        rgb_mask[:,:,2][t]=colors[c-1][2]
    return rgb_mask


def draw_boundary(annotations, offset=100):

    annotations = list(chain(*[annotations[f] for f in annotations]))
    coords = list(chain(*annotations))
    boundaries = list(map(lambda x: (min(x)-offset, max(x)+offset), list(zip(*coords))))
   
    return boundaries


def oneHotToMask(onehot):
    nClasses =  onehot.shape[-1]
    idx = tf.argmax(onehot, axis=-1)
    colors = sns.color_palette('hls', nClasses)
    multimask = tf.gather(colors, idx)
    multimask = np.where(multimask[:,:,:]==colors[0], 0, multimask[:,:,:])

    return multimask


#can we sample and return a new patching object
def sample_patches(patch,n,replacement=False):
    
    if replacement:
        patches=random.choice(patch._patches,n)
    else:
        patches=random.sample(patch._patches,n)

    new_patch =  Patch(patch.slide,
                       patch.size,
                       patch.mag_level=0,
                       patch.border,  
                       patch.step):

    new_patch.patches=patches
    return new_patches




