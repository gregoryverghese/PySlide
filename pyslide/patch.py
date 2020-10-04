#!/usr/local/share python
# -*- coding: utf-8 -*-

'''
patch.py
'''


import numpy as np
import cv2
import sys
import openslide
from utilities import draw_boundary
import itertools


class Patching():
    def __init__(self, slide, annotations, size=(256, 256), mag_level=4, boundaries=None, sparse_annotations=False):

        self.slide = slide
        self.annotations = annotations
        self.mag_level = mag_level
        self.boundaries = boundaries
        self.step = 64
        self.size = size
        self.sparse_annotations = sparse_annotations
        self.mag_factor = 16 
        self.number = None
        self.patches = []
        self.masks = []


    def __call__(self):
        
        print('what the hell are we doing')
        dim = self.slide.dimensions
        
        slide_mask = np.zeros((dim[1], dim[0]), dtype=np.uint8)

        for k in self.annotations:
            v = self.annotations[k]
            v = [np.array(a) for a in v]
            cv2.fillPoly(slide_mask, v, color=k)

        if not self.boundaries:
           x, y = self.slide.dimensions
           self.boundaries = [(0, x), (0, y)]

        elif self.boundaries=='draw':
            border = draw_boundary(self.annotations)
            x_min,x_max,y_min,y_max = list(itertools.chain(*border))

        for x in range(border[0][0], border[0][1], self.step*self.mag_factor):
            for y in range(border[1][0], border[1][1], self.step*self.mag_factor):
                self.patches.append({'x':x,'y':y})
                self.masks.append(slide_mask[x:x+self.size[0], y:y+self.size[1]])

        print('original number of patches', len(self.patches))
        if not self.sparse_annotations:
            index  = [i for i in range(len(self.masks)) if len(np.unique(self.masks[i])) > 1]
            patches = [self.patches[i] for i in index]
        

    def extract_patches(self):
            
        for p in self.patches:
            patch = p.read_region((p['x'],p['y'], self.mag_level, (self.size[0],self.size[1]))
            yield patch



    def sample_patches(self):
        pass
        

    def compute_class_weights(self):
        pass    
        #getclasslabels
        #calculate frequencies
        

    def compute_pixel_weights(self):
        #getnumberofpixels according to script
        pass








