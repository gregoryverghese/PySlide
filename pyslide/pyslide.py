#!/usr/bin/env python3

'''
pyslide.py: contains class Slide wrapped around openslide.Openslide to
load and annotate the whole slide images. Contains annotations class to
load annotations from json or xml format
'''

import sys
import os
import json
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
from openslide import OpenSlide
from itertools import chain
import operator as op


#__author__=='Gregory Verghese'
#__email__=='gregory.verghese@gmail.com'


class Slide(OpenSlide):
    """
    WSI object that enables annotation overlay

    wrapper around openslide.OpenSlide class loads WSIs 
    and provides additional functionality to generate 
    masks and mark with user loaded annotations

    Attributes:
        _slide_mask: ndarray mask representation
        dims: dimensions of WSI
        name: string name
        draw_border: boolean to generate border based on annotations
        _border: list of border coordinates [(x1,y1),(x2,y2)] 
    """

    MAG_fACTORS={0:1,1:2,3:4,4:16,5:32}

    def __init__(self, filename, draw_border=False, 
                 annotations=None, annotations_path=None):
        super().__init__(filename)
        
        if annotations_path is not None:
            ann=Annotations(annotations_path,labels)
            self.annotations=ann.generate_annotations()
        else:
            self.annotations=annotations
        
        self._slide_mask=None
        self.dims = self.dimensions
        self.name = os.path.basename(filename)[:-4]
        self.draw_border=draw_border
        self._border=None

    @property
    def border(self):
        return self._border

    @border.setter
    def border(self,value):
        #Todo: if two values we treat as max_x and max_y
        assert(len(value)==4)

    @property
    def draw_border():
        return self.draw_border

    @draw_border.setter 
    def draw_border(self, value):
        
        if value:
            self._border=self.draw_border()
            self.draw_border=value
        elif not value:
            self._border=[[0,self.dims[0]],[0,self.dims[1]]]
            self.draw_border=value
        else:
            raise TypeError('Boolean type required')
        
    
    def slide_mask(self, size=None):
        """
        generates mask representation of annotations

        Args:
            size: tuple of size dimensions for mask
        Returns:
            self._slide_mask: ndarray mask

        """

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
        """
        generate annotations object based on json or xml
        
        Args:
            path: path to json or xml annotation files
            file_type: xml or json
        Returns:
            self.annotations: dictionary of annotation coordinates
        """

        annotations_obj=Annotations(path, file_type)
        self.annotations = annotations.generate_annotations(labels)

        return self.annotations


    @staticmethod   
    def resize_border(dim, factor=1, threshold=None, operator='=>'):
        """
        resize and redraw annotations border - useful to cut out 
        specific size of WSI and mask

        Args:
            dim: dimensions
            factor: border increments
            threshold: min/max size
            operator: threshold limit 

        Returns:
            new_dims: new border dimensions [(x1,y1),(x2,y2)]

        """

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
        """
        generate border around annotations on WSI

        Args:
            space: border space
        Returns: 
            self._border: border dimensions [(x1,y1),(x2,y2)]
        """

        coordinates = list(chain(*[self.annotations[a] for a in 
                                   self.annotations]))
        coordinates=list(chain(*coordinates))
        f=lambda x: (min(x)-space, max(x)+space)
        self._border=list(map(f, list(zip(*coordinates))))

        return self._border


    def generate_region(self, mag=0, x=None, y=None, x_size=None, y_size=None, 
                        scale_border=False, factor=1, threshold=None, operator='=>'):
        """
        extracts specific regions of the slide

        Args:
            mag: magnfication level 1-8
            x: 
            y:
            x_size: x dim size
            y_size: y dim size
            scale_border: resize border
            factor: increment for resizing border
            threshold: limit for resizing border
            operator: operator for threshold
        Returns:
            region: ndarray image of extracted region
            mask: ndarray mask of annotations in region

        """

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
        """
        save thumbnail of slide in image file format
        Args:
            path:
            size:
            mask:
        """

        if mask:
            cv2.imwrite(path,self._slide_mask)
        else: 
            image = self.get_thumbnail(size)
            image = image.convert('RGB')
            image = np.array(image)
            cv2.imwrite(path,image)





class Annotations():

    """
    returns dictionary of coordinates of ROIs
    
    reads annotation files in either xml and json format
    and returns a dictionary containing x,y coordinates
    for each region of interest in the annotation

    Attributes:
        path: string path to annotation file
        annotation_type: file type
        labels: list of ROI names ['roi1',roi2']
        _annotations: dictonary with return files
                      {roi1:[[x1,y1],[x2,y2],...[xn,yn],...roim:[]}
    """
    def __init__(self, path, annotation_type=None,labels=None):
        self.path=path 
        self.annotation_type = annotation_type
        self.labels = labels
        self._annotations=None


    @property
    def class_key(self):
        if self.labels is not None:
            class_key={l:i for i, l in enumerate(self.labels)}
        return class_key


    def generate_annotations(self):
        """
        calls appropriate method for file type
        Returns:
            annotations: dictionary of coordinates
        """

        if self.annotation_type in ['imagej','xml']:
            annotations=self._xml()
        elif self.annotation_type=='json':
            annotations=self._json()
        elif self.path.endswith('json'):
            annotations=self._json()
        elif self.path.endswith('xml'):
            annotations=self._xml()
        else:
            raise ValueError('requires xml or json file')
        
        if self.labels is not None:
            #annotations=self.filter_labels(annotations)
            pass

        return annotations


    def filter_labels(self):
        """
        remove labels from annotations
        
        Returns:
            annotations: filtered dict of coordinates
        """

        keys = list(self.annotations.keys())
        for k in keys:
            if k not in self.labels:
                del self.annotations[k]
        return self.annotations       


    def _imageJ(self):
        """
        parses xml files

        Returns:
            annotations: dict of coordinates
        """

        tree=ET.parse(self.path)
        root=tree.getroot()
        regions = {n.attrib['Name']: n[1].findall('Region') for n in root}

        if self.labels is None:
            self.labels=list(regions.keys())
        
        annotations={}
        for l in regions:
            region_dict={}
            for i,c in enumerate(regions[l]):
                verts=c[1].findall('Vertex')
                verts=[(x.attrib['X'], x.attrib['Y']) for x in verts]
                region_dict[i]=[(int(float(x)),int(float(y))) for x,y in verts]
            annotations[l]=region_dict 
        
        annotations = {self.class_key[k]: list(v.values()) for k,v in annotations.items()}
        self._annotations=annotations

        return annotations


    def _asap(self):

        tree=ET.parse(path)
        root=tree.getroot()
        ns=root[0].findall('Annotation')
        labels=list(root.iter('Annotation'))
        labels=list(set([i.attrib['PartOfGroup'] for i in labels]))
        annotations={l:[] for l in labels}
        for i in ns:
            coordinates=list(i.iter('Coordinate'))
            coordinates=[(float(c.attrib['X']),float(c.attrib['Y'])) for c in coordinates]
            coordinates=[(round(c[0]),round(c[1])) for c in coordinates]
            label=i.attrib['PartOfGroup']
            annotations[label]=annotations[label]+coordinates
        self._annotations=annotations
        return annotations
            

    def _json(self):
        """
        parses json file

        Returns:
            annotations: dict of coordinates
        """

        with open(self.path) as json_file:
            json_annotations=json.load(json_file)
        
        if self.labels is None:
            self.labels=list(json_annotations.keys())

        annotations = {self.class_key[k]: [[int(i['x']), int(i['y'])] for i in v2] 
                       for k, v in json_annotations.items() for v2 in v.values()}

        self._annotations=annotations
        return annotations


    def _dataframe(self):
        pass


    def _csv(self):
        pass


    def df(self):
        """
        returns dataframe of annotations

        """
        key={v:k for k,v in self.class_key.items()}
        labels=[[l]*len(self._annotations[l]) for l in self._annotations.keys()]
        labels=chain(*labels)
        labels=[key[l] for l in labels]
        x_values=[xi[0] for x in list(self._annotations.values()) for xi in x]  
        y_values=[yi[1] for y in list(self._annotations.values()) for yi in y]
        df=pd.DataFrame({'labels':labels,'x':x_values,'y':y_values})

        return df


    def save(self,save_path):
        """
        save down annotations in csv file
        Args:
            save_path:string save path
        """
        self.df().to_csv(save_path)



