#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
pyslide.py
'''

import sys
import os

import numpy as np
import cv2
from openslide import OpenSlide
from itertools import chain
import operator as op


class Slide(OpenSlide):

    MAG_fACTORS={0:1,1:2,3:4,4:16,5:32}

    def __init__(self, filename, border,
                 annotations=None,annotations_path=None;):
        super().__init__(filename)
        
        if annotations_path not None:
            ann=Annotations(annotations_path,labels)
            self.annotations=ann.generate_annotations()
        else:
            self.annotations=annotations
        
        self._slide_mask=None
        self.dims = self.dimensions
        self.name = os.path.basename(filename)[:-4]

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
        
    
    def oneHotToMask(onehot):

        nClasses =  onehot.shape[-1]
        idx = tf.argmax(onehot, axis=-1)
        colors = sns.color_palette('hls', nClasses)
        multimask = tf.gather(colors, idx)
        multimask = np.where(multimask[:,:,:]==colors[0], 0, multimask[:,:,:])

        return multimask


    def slide_mask(self, size=None):
        
        x, y = self.dims[0], self.dims[1]
        slide_mask=np.zeros((y, x, 3), dtype=np.uint8)
        
        for k in self.annotations:
            v = self.annotations[k]
            v = [np.array(a) for a in v]
            cv2.fillPoly(slide_mask, v, color=k)

        if size is not None:
            slide_mask=cv2.resize(slide_mask, size)
             
        self._slide_mask=slide_mask
        
        return self._slide_mask


    @staticmethod 
    def generate_annotations(labels,path,file_type):

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
            image = image.convert('RGB')
            image = np.array(image)
            cv2.imwrite(path,image)



class Annotations():
    def __init__(self, paths, file_type=None):
        self.paths=paths 
        self.type = file_type
        self.labels = labels

    def generate_annotations(self):

        if file_type in ['imagej','xml']:
            annotations=self._xml_file()
        elif paths.endswith('json'):
            annotations=self._json_file():
        
        annotations=filter_labels(annotations)
        return annotations


    def filter_labels(self,annotations):
        keys = list(json_annotations.keys())
        for k in keys:
            if k not in self.labels:
                del annotations[k]
        return annotations       


    def _xml_file(self):
            
        annotations={}
        pixelSpacing = float(root.get('MicronsPerPixel'))
        for l in labels:
            l_dict={}
            for i,c in enumerate(labels[l]):
                verts=c[1].findall('Vertex')
                verts=[(x.attrib['X'], x.attrib['Y']) for x in verts]
                l_dict[i]=[(int(float(x)),int(float(y))) for x,y in verts]
                annotations[l]=l_dict
        return annotations


    def _json_file(self):
        
        with open(self.paths) as json_file:
            json_annotations=json.load(json_file)

        annotations = {labels[k]: [[[int(i['x']), int(i['y'])] for i in v2] for
                       k2, v2 in v.items()]for k, v in json_annotations.items()}
        return annotations



    def _dataframe(self):
        pass


    def _csv(self):
        pass

