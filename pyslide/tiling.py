#!/usr/local/share python
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import openslide


class Tiling():
   def __init__():
       pass

   def extract_patches(self, slide, boundaries, annotations, offset, masks=false):

       x1,x2,y1,y2 = boundaries
       #ToDo: offset x1, y1 user defined int(self.tileDim*0.5*self.magFactor)
       for w in range(boundaries[0][0], boundaries[0][1],self.step*self.mag_level):
           for h in range(boundaries[1][0], boundaries[1][1], self.step*self.mag_level):
               Patch(_).extract((x1, y1))
               
               if mask:
                   mask = slide.mask
                   mask = img[h:h+self.x_size, w:w+self.y_size]
                   masks.append(mask)
                   patches.append(patch)

                   return patches, masks

               patches.append(patch)

        return patches

    

   def generate_masks():
       pass





