#!/usr/local/share python
# -*- coding: utf-8 -*-

'''
patch.py
'''


import numpy as np
import cv2
import sys
import openslide


def draw_boundary(annotations, offset=100):

    annotations = list(chain(*[annotations[f] for f in annotations]))
    coords = list(chain(*annotations))
    boundaries = list(map(lambda x: (min(x)-offset, max(x)+offset), list(zip(*coords))))
    #print('boundaries: {}'.format(boundaries))
    return boundaries


class Patch():
    def __init__(self):
        self.x = x
        self.y = y
        self.size = size
        self.mag_level = mag_level
        

    def extract(self, slide):
        #return np.array(slide.get_region((self.x, self.y), self.mag_level, self.size))
        pass 
    
    def get_origin(self):
        #return (self.x, self.y)
        pass


    def get_size(self):
        #return self.size
        pass


    def patch_intensity(self):
        #return np.mean()
        pass


class Patching():
    def __init__(self, slide, mag_level):
        self.slide = slide
        self.mag_level = mag_level
        
        
    def extract_patches(self, size, annotations=None, step=0, boundaries=False, only_annotations=False):
       
       patches=[]
       
       #check whether user has passed boundaries or define using annotation
       #extremis
       if not boundaries:
           x, y = slide.dimensions
           boundaries = [(0, x), (0, y)]
           print(boundaries)

       elif boundaries=='draw':
           boundaries = draw_boundary(annotations)

       #x1,x2,y1,y2 = boundaries
       #ToDo: offset x1, y1 user defined int(self.tileDim*0.5*self.magFactor)

       for x in range(boundaries[0][0], boundaries[0][1],step*self.mag_level):
           for y in range(boundaries[1][0], boundaries[1][1], step*self.mag_level):
               #Patch(_).extract((x1, y1))
               patches.append(slide.read_region((x, y), self.mag_level, size).convert('RGB'))

       #we might want to focus only on patches containing annotations
       elif only_annotations:
           for x in range(boundaries[0][0], boundaries[0][1],step*self.mag_level):
               for y in range(boundaries[1][0], boundaries[1][1], step*self.mag_level):
                   for a in list(annotations.values())
                       p = Path(a)
                       contains = p.contains_point([w, h])
                       if contains==True:
                           patches.append(slide.read_region((x, y)), self.mag_level,size).convert('RGB')
                           break

       return patches

    
    def generate_masks(self):
        pass


slide = openslide.OpenSlide('U_100188_10_X_HIGH_10_L1.ndpi')
p=Patching(slide, 4)
patches=p.extract_patches((250,250), step=125)
print(len(patches), sys.getsizeof(patches))
print(patches[0])
cv2.imwrite('test.png', np.array(patches[10000]))





