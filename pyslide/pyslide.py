#!/usr/local/share python
# -*- coding: utf-8 -*-

'''
pyslide.py
'''
import numpy as np
import cv2
import sys
from openslide import OpenSlide
from itertools import chain
import operator as op


class Slide(OpenSlide):

    MAG_fACTORS={0:1,1:2,3:4,4:16,5:32}

    def __init__(self, filename, border, annotations=None):
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
        
        colors=[(255,0,0),(0,255,0),(0,0,255)]
        x, y = self.dims[0], self.dims[1]
        slide_mask=np.zeros((y, x, 3), dtype=np.uint8)
        
        for k in self.annotations:
            v = self.annotations[k]
            v = [np.array(a) for a in v]
            cv2.fillPoly(slide_mask, v, color=colors[k])

        if size is not None:
            slide_mask=cv2.resize(slide_mask, size)
             
        self._slide_mask=slide_mask
        
        return self._slide_mask



    def generate_annotations(self,labels,path,file_type):

        annotations_obj=Annotations(path, file_type)
        self.annotations = annotations.generate_annotations(labels)

        return self.annotations


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

    
    def save(self, path, size=(2000,2000), mask=False):

        if mask:
            cv2.imwrite(path,self._slide_mask)
        else: 
        image = self.get_thumbnail(size)
        if mask:
            cv2.imread(path,image)





######################################################################3

            








