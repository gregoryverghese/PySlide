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
        self._number = None
        self._patches = []
        self._masks = []
        self._slide_mask = None


    def __call__(self):
        
        dim = self.slide.dimensions
        class_no = []

        slide_mask = np.zeros((dim[1], dim[0]), dtype=np.uint8)

        for k in self.annotations:
            v = self.annotations[k]
            v = [np.array(a) for a in v]
            cv2.fillPoly(slide_mask, v, color=k)
        
        self._slide_mask = slide_mask

        if not self.boundaries:
           x, y = self.slide.dimensions
           self.boundaries = [(0, x), (0, y)]

        elif self.boundaries=='draw':
            border = draw_boundary(self.annotations)
            x_min,x_max,y_min,y_max = list(itertools.chain(*border))
        
        step = self.step*self.mag_factor

        for x in range(border[0][0], border[0][1], step):
            for y in range(border[1][0], border[1][1], step):

                self.patches.append({'x':x,'y':y})
                mask = slide_mask[x:x+self.size[0],y:y+self.size[1]]
                classes = dict(zip(*np.unique(mask,return_counts=True)))
                self.masks.append({'x':x, 'y':y, 'classes':classes})


        print('original number of patches', len(self.patches))
        if not self.sparse_annotations:
            index  = [i for i in range(len(class_no)) if class_no[i] > 1]
            patches = [self.patches[i] for i in index]



    @property
    def masks(self):
        return self._masks


    @masks.setter
    def masks(self):
        pass        


    @property
    def patches(self):
        return self._patches


    @patches.setter
    def patches(self):
        pass


    @property
    def slide_mask(self, size):
        mask = self._slide_mask
        mask = cv2.resize(mask, size)
        return mask
        
        
    def extract_patches(self):
            
        for p in self.patches:
            patch = p.read_region((p['x'],p['y'], self.mag_level, (self.size[0],self.size[1])))
            yield patch


    def extract_masks(self):
        
        for p in self.patches:
            x = p['x']
            y = p['y']
            mask = self._slide_mask[x:x+self.size[0], y:y+self.size[1]]
            yield mask


    def sample_patches(self):
        pass
        
        

    def compute_class_weights(self):
        #labels = self.patches.labels
        pass
        
        
    def compute_pixel_weights(self):

        for m in self.masks:
            labels = m.reshape(-1)
            classes = np.unique(labels)
            weightDict = {c:0 for c in range(numClasses)}
            classWeights = class_weight.compute_class_weight('balanced', classes, labels)

            weightKey = list(zip(classes, classWeights))
            for k, v in weightKey:
                weightDict[k]=v

            values=list(weightDict.values())
            weights.append(list(values))

        finalWeights = list(zip(*weights))
        averageWeights = [np.mean(np.array(w)) for w in finalWeights]









