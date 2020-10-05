#from patch import Patching
from utilities import getRegions 
import openslide
from patch import Patching
import sys
import cv2
import numpy as np

classKey = {'SINUS':1}

annotations = getRegions('U_100188_10_X_HIGH_10_L1.xml')

keys = annotations.keys()
for k in list(keys):
    if k not in classKey:
        del annotations[k]

annotations = {classKey[k]: [v2['coords'] for k2, v2 in v.items()] for k,v in annotations.items()}


print(annotations.keys())





slide = openslide.OpenSlide('U_100188_10_X_HIGH_10_L1.ndpi')
p=Patching(slide, annotations, boundaries='draw')
p()
patches = p.patches
masks = p.masks

slide_mask = p.slide_mask(size=(1000,1000))

cv2.imwrite('test.png', slide_mask)



