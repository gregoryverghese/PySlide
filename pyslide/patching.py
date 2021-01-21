#!usr/bin/env python3

import os

import numpy as np
import openslide
import cv2
from matplotlib.path import Path

from PySlide.pyslide import Slide


class Patching():

    MAG_FACTORS={0:1,1:2,3:4,4:16,5:32}

    def __init__(self, slide, annotations, size=(256, 256), mag_level=0,
            border=None, mode=False):
    
        super().__init__()
        self.slide=slide 
        self.mag_level = mag_level
        self.size = size
        #self.mode = mode 
        self._number = None
        self._patches = []
        self._masks = []
        self._class_no = []
        self.__magfactor=Patching.MAG_FACTORS[self.mag_level]
    

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
    def slide_mask(self):
        return self.slide._slide_mask


    @staticmethod
    def patching(step,xmin,xmax,ymin,ymax):

        for x in range(xmin,xmax, step):
            for y in range(ymin,ymax,step): 
                yield x, y

    @property
    def slide_mask(self):
        return self.slide._slide_mask


    #TODO discard patches at borders that do not match size
    def generate_patches(self,step, mode='sparse'):
   
        step=step*self.__magfactor
        #mask=self._slide_mask()
        #mask=self.slide._slide_mask

        xmin, xmax = self.slide.border[0][0], self.slide.border[0][1]
        ymin, ymax = self.slide.border[1][0], self.slide.border[1][1]

        for x, y in self.patching(step, xmin,xmax,ymin,ymax):

            self.patches.append({'x':x,'y':y})
            mask = self.slide._slide_mask[y:y+self.size[0],x:x+self.size[1]]

            classes = len(np.unique(mask)) if mode=='focus' else 1
            self.masks.append({'x':x, 'y':y, 'classes':classes})

        if mode=='focus':
            self.focus()

        return len(self._patches)


    def focus(self):

        index=[i for i in range(len(self._patches)) if
               self._masks[i]['classes'] >1]

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
    def within(self, boundaries=None):

        if boundaries is None:
            boundaries=self.slide.border

        path = Path(boundaries)
        f = lambda x: p.contains([x['x'],x['y']])
        self_.patches=list(filter(f, self._patches))
    
        return self_patches
    
    
    def extract_patch(self, x=None, y=None):
        patch=self.slide.read_region((x,y),self.mag_level,(self.size[0],self.size[1]))
        patch=np.array(patch.convert('RGB'))
        return patch


    def extract_patches(self):
        for p in self._patches:
            patch=self.extract_patch(p['x'],p['y'])
            yield patch,p['x'],p['y']
    
       
    def extract_mask(self, x=None, y=None):
        mask=self.slide_mask[y:y+self.size[0],x:x+self.size[1]]
        return mask 
    
    
    def extract_masks(self):
        for p in self._patches:
            print(p['x'],p['y'])
            mask=self.extract_mask(p['x'],p['y'])
            yield mask,p['x'],p['y']


    def save(self, path, mask=False, x=None, y=None):
        
        if (x and not y) or (not x and y):
            raise ValueError('missing value for x or y')

        patchpath=os.path.join(path,'images')
        try:
            os.mkdir(patchpath)
        except OSError as error:
            print(error)

        maskpath=os.path.join(path,'masks')
        try:
            os.mkdir(os.path.join(maskpath))
        except OSError as error:
            print(error)

        if not (x and y) is None:
            patch=self.extract_patch()
            patchpath=os.path.join(patchpath,self.slide.name+'_'+str(x)+'_'+str(y)+'.png')
            patchstatus=cv2.imwrite(patchpath,patch)
            if mask:
                mask,x,y=self.extract_mask()
                maskpath=os.path.join(maskpath,self.slide.name+'_'+str(x)+'_'+str(y)+'.png')
                maskstatus=cv2.imwrite(maskpath,mask)
                return patchstatus and maskstatus
            return patchstatus 

        for patch,x,y in self.extract_patches():
            patchpath=os.path.join(patchpath,self.slide.name+'_'+str(x)+'_'+str(y)+'.png')
            patchstatus=cv2.imwrite(patchpath,patch)
            if mask:
                mask,x,y=self.extract_masks()
                maskpath=os.path.join(maskpath,self.slide.name+'_'+str(x)+'_'+str(y)+'.png')
                maskstatus=cv2.imwrite(maskpath,mask)
                return patchstatus and maskstatus  
            return patchstatus   

       




        





'''
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
        for p in self.extract_patches()
            self.image[ynew:ynew+ysize,xnew:xnew+xsize,0]=p[:,:,0]
            self.image[ynew:ynew+ysize,xnew:xnew+xsize,1]=p[:,:,1]
            self.image[ynew:ynew+ysize,xnew:xnew+xsize,1]=p[:,:,1]
        
        return image
'''
