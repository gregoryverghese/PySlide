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
import itertoolsi
import operator as op


class Slide(OpenSlide):
    def __init__(self, annotations):
        super().__init__()
        self.annotations=annotations
        self._slide_mask=None
       

    @property
    def border(self):
        return self._border


    @setter.border
    def border(self, value):

        if value=='draw':
            self.border = draw_boundary(self.annotations)
        elif value=='fullsize':
            self.border = [[0,self.dims[0]],[0,self.dims[0][1]]
        else:
            self.border = value
          

    def slide_mask(self, size=None):
        
        x, y = self.dims[0], self.dims[1])
        slide_mask=np.zeros((x, y), dtype=np.uint8)

        for k in self.annotations:
            v = self.annotations[k]
            v = [np.array(a) for a in v]
            cv2.fillPoly(slide_mask, v, color=k)

        if size not None:
            cv2.resize(slide_mask, size)
        
        self._slide_mask=slide_mask

        return slide_mask


    @staticmethod   
    def resize_border(self, dim, factor, threshold, operator):

        operator_dict={'>':op.gt,'=>':op.ge,'<':op.lt,'=<':op.le}
        
        multiples = [factor*i for i in range(100000)]
        multiples = [m for m in multiples if op(m,threshold)]
        diff = list(map(lambda x: abs(dim-x), multiples))
        new_dim = multiples[diff.index(min(diff))]

    return new_dim


    def draw_border(self, space=100)
        
        coordinates=list(chain(*self.annotations))
        coordiantes=list(chain(*annotations))
        f=lambda x: (min(x)-space, max(x)+space)
        self.border=list(map(f, list(zip*coordinates)))

        return self.border


    def generate_region(self, x=None, y=None,  x_size=None, y_size=None,
                        scale_border=None):

        if x is None:
            self.draw_border()
            x, y = self.border
        
        if scale_border:
            x = resize_border(x, factor, threshold, operator)
            y = resize_border(y, factor, threshold, operator)

        region=slide.read_region((x,y),mag,size)
        mask=self.slide_mask[x:x+xsize,y:y+ysize]

        return region


    def calculate_mean(self)
        pass 


    def calculate_std(self)
        pass


    def calcuate_weights()
        pass


######################################################################3

class Patching(Slide):

    MAG_fACTORS={0:1,1:2,3:4,4:16,5:32}

    def __init__(self, slide, annotations, size=(256, 256), mag_level=0,
            border=None, mode=False):
        
        super()__init__()
        self.mag_level = mag_level
        self.border = border 
        self.size = size
        self.mode = mode 
        self._number = None
        self._step = step 
        self._patches = []
        self._masks = []
        self._slide_mask = None
        self._class_no = []
        self.__magfactor=mag_factors[self.mag_level]
                

    @property
    def masks(self):
        return self._masks
     

    @property
    def patches(self):
        return self._patches


    @property
    def annotations(self):
        return _self.annotations

    
    @property 
    def step(self):
        return self._step


    @step.setter():
    def step(self, value):
        step=step*mag_factors[self.mag_level]
        self._step=step

    
   def patching(self):

        xmin, xmax = border[0][0], border[0][1]
        ymin, ymax = border[1][0], border[1][1]
        for x in range(xmin,xmax, self.step):
            for y in range(ymin,ymax,self.step): 
                yield x, y

    #TODO discard patches at borders that do not match size
    def generate_patches(self):
        
        mask=self.slide_mask()

        for p in patching:
            self.patches.append({'x':x,'y':y})
            mask = self.slide_mask[y:y+self.size[0],x:x+self.size[1]]
            classes = dict(zip(*np.unique(mask,return_counts=True)))
            self._class_no.append(len(classes))
            self.masks.append({'x':x, 'y':y, 'classes':classes})

        if self.mode=='focus':
            self.contains()

        return len(self._patches)


    def focus(self):

        index  = [i for i in range(len(self._class_no)) 
                  if self._class_no[i] > 1]

        self._patches = [self.patches[i] for i in index]

        return len(self._patches)

    
    #TODO:check my filter method
    @staticmethod 
    def contains(verts):

        xx,yy=np.meshgrid(np.arange(300),np.arange(300))
        xx,yy=xx.flatten(),yy.flatten()
        verts=np.stack([x,y]).T
        p=Path(verts)
        mask=p.contains_points(verts)
        num=(tolerance*grid.shape[0])
        x = len(grid[grid==True])
        return verts


    #Do we want to use filtering based on orign point
    #or do we want to filter based on all points within patch
    def within(self, boundaries=self._boundaries):

        path = Path(self.boundaries)
        f = lambda x: p.contains([x['x'],x['y']])
        self_.patches=list(filter(f, self._patches))
     
        return self_patches
        
                
    def extract_patch(self, x=None, y=None):
        patch=self.slide.read_region(x,y,self.mag_level,(self.size[0],self.size[1])
        return p


    def extract_patches(self):
        for p in self._patches:
            patch=extract_patch(p['x'],p['y'])
            yield patch
        
    
    def extract_mask(self, x=None, y=None):
        mask=slide_mask[y:y+self.size[0],x:x+self.size[1]]
        return mask 
        
        
    def extract_masks(self):
        for p in self._patches:
            mask=self.extract_mask(p['x'].p['y']
            yield mask
        





####################################################################


class Stitching():
    def __init__(self, _patches):
        super().__init__()
        self.x = self.patches.dims[0]
        self.y = self.patches.dims[1]
        self.image = np.zeros((self.x, self.y))

    @property
    def slide(self):
        return self._slide
 

    def stitch():
        
        temp=np.zeros((int(h), int(w, 3))
        for p in extract_patches():
            self.image[ynew:ynew+ysize,xnew:xnew+xsize,0]=p[:,:,0]
            self.image[ynew:ynew+ysize,xnew:xnew+xsize,1]=p[:,:,1]
            self.image[ynew:ynew+ysize,xnew:xnew+xsize,1]=p[:,:,1]
        
        return image


            






