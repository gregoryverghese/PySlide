#from patch import Patching
from utilities import getRegions 
import openslide
from patch import Patching
import sys
import cv2
import numpy as np
import os

classKey = {'SINUS':1}

annotations = getRegions('U_100188_10_X_HIGH_10_L1.xml')

keys = annotations.keys()
for k in list(keys):
    if k not in classKey:
        del annotations[k]

annotations = {classKey[k]: [v2['coords'] for k2, v2 in v.items()] for k,v in annotations.items()}

print(annotations.keys())
slide = openslide.OpenSlide('U_100188_10_X_HIGH_10_L1.ndpi')
p=Patching(slide, annotations, boundaries='draw',mode='focus')
p()
#patches = p.patches
#masks = p.masks


patches = p.extract_patches()
masks = p.extract_masks()

for i, p in enumerate(patches):
    p=np.array(p.convert('RGB'))
    cv2.imwrite(os.path.join('patches', str(i) + '.png'), p)

for i, m in enumerate(masks):
    cv2.imwrite(os.path.join('masks', str(i) + '.png'), m)



'''
for i, _ in enumerate(p.patches):
    patch = p.extract_patch()
    mask = p.extract_mask()
    patch = np.array(patch)
    mask = np.array(mask)
    #print(np.unique(mask))
    cv2.imwrite(os.path.join('patches', str(i) + '.png'), patch)
    cv2.imwrite(os.path.join('masks', str(i) + '.png'), mask)


#x = p.slide_mask*255

#print(np.unique(x))
#cv2.imwrite('test.png', x)
'''


