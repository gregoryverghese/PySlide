#!/usr/local/share python3
# -*- coding: utf-8 -*-

'''
measure.py
'''

import os

import cv2
import numpy as np
import openslide



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
        erageWeights = [np.mean(np.array(w)) for w in finalWeights]
