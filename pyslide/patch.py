#!/usr/local/share python
# -*- coding: utf-8 -*-

'''
patch.py
'''
import numpy as np
import cv2
import sys
from openslide import OpenSlide
from itertools import chain
import operator as op


class Slide(OpenSlide):

    MAG_fACTORS={0:1,1:2,3:4,4:16,5:32}

    def __init__(self, filename, annotations, border):
        super().__init__(filename)
        self.annotations=annotations
        self._slide_mask=None
        self.dims = self.dimensions

        if border=='draw':
            self._border=self.draw_border()
        elif border=='fullsize':
            self._border=[[0,self.dims[0]],[0,self.dims[1]]]
       
    @property
    def border(self):
        return self._border


    @border.setter 
    def border(self, value):
        
        if value=='draw':
            self._border = self.draw_border()
        elif value=='fullsize':
            self._border = [[0,self.dims[0]],[0,self.dims[1]]]
        else:
            pass
            #TODO need to raise an exception
          

    def slide_mask(self, size=None):
        
        x, y = self.dims[0], self.dims[1]
        slide_mask=np.zeros((x, y), dtype=np.uint8)

        for k in self.annotations:
            v = self.annotations[k]
            v = [np.array(a) for a in v]
            cv2.fillPoly(slide_mask, v, color=k)

        if size is not None:
            slide_mask=cv2.resize(slide_mask, size)
        
        self._slide_mask=slide_mask

        return slide_mask


    @staticmethod   
    def resize_border(dim, factor=1, threshold=None, operator='=>'):
        
        if threshold is None:
            threshold=dim

        operator_dict={'>':op.gt,'=>':op.ge,'<':op.lt,'=<':op.lt}
        operator=operator_dict[operator] 
        multiples = [factor*i for i in range(100000)]
        multiples = [m for m in multiples if operator(m,threshold)]
        diff = list(map(lambda x: abs(dim-x), multiples))
        new_dim = multiples[diff.index(min(diff))]
       
        return new_dim


    #TODO: function will change with format of annotations
    #data structure accepeted
    def draw_border(self, space=100):

        coordinates = list(chain(*[self.annotations[a] for a in 
                                   self.annotations]))
        coordinates=list(chain(*coordinates))
        f=lambda x: (min(x)-space, max(x)+space)
        self._border=list(map(f, list(zip(*coordinates))))

        return self._border


    def generate_region(self, mag=0, x=None, y=None,  x_size=None, y_size=None,
                        scale_border=False, factor=1, threshold=None,
                        operator='=>'):

        if x is None:
            self.draw_border()
            x, y = self.border
        
        x_min, x_max=x
        y_min, y_max=y

        x_size=x_max-x_min
        y_size=y_max-y_min

        #Adjust sizes - treating 0 as base
        #256 size in mag 0 is 512 in mag 1
        x_size=int(x_size/Slide.MAG_fACTORS[mag])
        y_size=int(y_size/Slide.MAG_fACTORS[mag])
        
        if scale_border:
            x_size = Slide.resize_border(x_size, factor, threshold, operator)
            y_size = Slide.resize_border(y_size, factor, threshold, operator)
        
        print('x_size:{}'.format(x_size))
        print('y_size:{}'.format(y_size))

        region=self.read_region((x_min,y_min),mag,(x_size, y_size))
        mask=self.slide_mask()[x_min:x_min+x_size,y_min:y_min+y_size]

        return region, mask 



class Preprocessing():
    def __init__(self):
        self.masks = mask
        self.patches = patch
        self._weights = None

    @property
    def weights(self):
        return self_weights


    
    def image_std(image):

        channel_values = np.sum(image, axis=(0,1), dtype='float64')
        shape = image.shape

        channel_values_sq= no.sum(np.square(image-mean), axis=(0,1),
                                  dtype='float64')

        std = np.sqrt(channel_values_sq/pixel_num, dtype='float64')

        return std

        
    
    @staticmethod
    def image_mean(image):

        channel_values = np.sum(image, axis=(0,1), dtype='float64')
        shape = image.shape

        pixel_num = image_shape[0]*image_shape[1]
        mean=channel_values/pixel_num

        return mean, channel_values, pixel_num


    def calculate_mean(self):
        if channel:
            pass

        for p in self._patches:
            p=p.astype('float64')
            mean=image_mean

        return None


    def calculate_std(self):

        if channel:
            pass 

        for p in self._patches:
            p=p.astype('float64')
            std = image_std

        return None



    #TODO calculate inverse frequency of pixels and compare
    def calculate_weights(self, no_classes):
    
        total = {c:0 for c in range(num_classes)}

        for m in self.masks:
            labels = m.reshape(-1)
            classes = np.unique(labels, return_counts=True)

            pixel_dict = dict(list(zip(*classes))) 
    
            for k, v in pixel_dict.items():
                total[k] = total[k] + v 
        
        if num_classes==2:
            self._weights = total[0]/total[1]
        else:
            self._weights = [1/v for v in list(total.values())]

        return self_weights



######################################################################3

class Patching(Slide):

    MAG_fACTORS={0:1,1:2,3:4,4:16,5:32}

    def __init__(self, slide, annotations, size=(256, 256), mag_level=0,
            border=None, mode=False):
        
        super().__init__()
        self.slide=slide 
        self.mag_level = mag_level
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


    @step.setter
    def step(self, value):
        step=step*MAG_FACTORS[self.mag_level]
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
    def within(self, boundaries=None):

        if boundaries is None:
            boundaries=self.slide.border

        path = Path(boundaries)
        f = lambda x: p.contains([x['x'],x['y']])
        self_.patches=list(filter(f, self._patches))
     
        return self_patches
        
                
    def extract_patch(self, x=None, y=None):
        patch=self.slide.read_region(x,y,self.mag_level,(self.size[0],self.size[1]))
        return patch


    def extract_patches(self):
        for p in self._patches:
            patch=extract_patch(p['x'],p['y'])
            yield patch
        
    
    def extract_mask(self, x=None, y=None):
        mask=slide_mask[y:y+self.size[0],x:x+self.size[1]]
        return mask 
        
        
    def extract_masks(self):
        for p in self._patches:
            mask=self.extract_mask(p['x'],p['y'])
            yield mask
        


####################################################################

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

            






