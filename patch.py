#!/usr/local/share python
# -*- coding: utf-8 -*-

'''
patch.py
'''


import numpy as np
import cv2
import openslide



class Patch():
    def __init__():
        self.x = x
        self.y = y
        self.size = size
        self.mag_level = mag_level
        

    def extract(self, slide):
        return np.array(slide.get_region((self.x, self.y), self.mag_level, self.size))
        
    
    def get_origin(self):
        return (self.x, self.y)


    def get_size(self):
        return self.size


    def patch_intensity(self)
        return np.mean()


class Slide():
    def __init__():
        pass












