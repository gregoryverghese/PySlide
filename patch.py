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


class Patching():
    def __init__(self, slide, size=(256, 256), mag_level=0, boundaries=None, sparse_annotations=False):

        self.slide = slide
        self.mag_level = mag_level
        self.boundaries = boundaries
        self.size = size
        self.sparse_annotations = sparse_annotations

        self.number = None


    def extract_patches(self, annotations):
                
        #mask = slide.generate_mask()
        dim = self.slide.dimensions
        img = np.zeros((dim[1], dim[0]), dtype=np.uint8)
        masks = []
        patches = []

        for k in annotations:
            v = annotations[k]
            v = [np.array(a) for a in v]
            cv2.fillPoly(img, v, color=k)
 
        if not boundaries:
           x, y = slide.dimensions
           boundaries = [(0, x), (0, y)]
           print(boundaries)

        elif self.boundaries=='draw':
            border = draw_boundary(annotations)
            x_min,x_max,y_min,y_max = list(itertools.chain(*border))

        for x in range(border[0][0], border[0][1],step*self.mag_level):
            for y in range(border[1][0], border[1][1], step*self.mag_level):
                masks.append(img[h:h+self.size, w:w+self.size])
                patches.append(slide.read_region((x, y), self.mag_level, size).convert('RGB'))

        if not sparse_annotations:
            index  = [i for i in range(len(masks)) if np.unique(masks[i]) > 1]
            patches = [patches[i] for i in index]
            masks = [masks[i] for i in index]
         
        return masks, patches

    '''
    def extract_patches(self):

        if not boundaries:
            x, y = slide.dimensions
            boundaries = [(0, x), (0, y)]
        elif self.boundaries=='draw':
            border = draw_boundary(self.annotations)
            x_min,x_max,y_min,y_max = list(itertools.chain(*border))

        for x in range(border[0][0], border[0][1],step*self.mag_level):
            for y in range(border[1][0], border[1][1], step*self.mag_level):
                patches.append(slide.read_region((x, y), self.mag_level, size).convert('RGB'))

        if not sparse_annotations:
            index  = [i for i in range(len(masks)) if np.unique(masks[i]) > 1]
            patches = [patches[i] for i in index]

        return patches
    '''


    def sample_patches(self):
        pass
        

    def compute_class_weights(self):
        pass    
        #getclasslabels
        #calculate frequencies
        

    def compute_pixel_weights(self):
        #getnumberofpixels according to script
        pass








