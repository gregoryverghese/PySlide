#!usr/bin/env python3

"""
slide.py
"""

import os
import glob
import json
import xml.etree.ElementTree as ET

import numpy as np
import openslide
import cv2
import seaborn as sns
from matplotlib.path import Path
from openslide import OpenSlide
import pandas as pd
import seaborn as sns
from itertools import chain
import operator as op
from pyslide.utilities import mask2rgb


__author__='Gregory Verghese'
__email__='gregory.verghese@gmail.com'


class Slide(OpenSlide):
    """
    WSI object that enables annotation overlay wrapper around 
    openslide.OpenSlide class. Generates annotation mask.

    :param _slide_mask: ndarray mask representation
    :param dims dimensions of WSI
    :param name: string name
    :param draw_border: boolean to generate border based on annotations
    :param _border: list of border coordinates [(x1,y1),(x2,y2)]
    """
    MAG_fACTORS={0:1,1:2,2:4,3:8,4:16,5:32}

    def __init__(self,filename,mag=0,annotations=None,
                 annotations_path=None,labels=None,source=None):
        super().__init__(filename)

        self.mag=mag
        self.dims=self.dimensions
        self.name=os.path.basename(filename)
        self._border=None
        if annotations_path is not None:
            self.annotations=Annotations(annotations_path,source=source,
                                         labels=labels,encode=True)
        else:
            self.annotations=annotations


    @property
    def slide_mask(self):
       mask=self.generate_mask((2000,2000))
       mask=mask2rgb(mask)
       return mask


    def generate_mask(self, size=None):
        """
        Generates mask representation of annotations.

        :param size: tuple of mask dimensions
        :return: self._slide_mask ndarray. single channel
            mask with integer for each class
        """
        x, y = self.dims[0], self.dims[1]
        slide_mask=np.zeros((y, x), dtype=np.uint8)
        self.annotations.encode=True
        coordinates=self.annotations.annotations
        keys=sorted(list(coordinates.keys()))
        for k in keys:
            v = coordinates[k]
            v = [np.array(a) for a in v]
            cv2.fillPoly(slide_mask, v, color=k)
        if size is not None:
            slide_mask=cv2.resize(slide_mask, size)
        return slide_mask


    @staticmethod
    def resize_border(dim, factor=1, threshold=None, operator='=>'):
        """
        Resize and redraw annotations border. Useful to trim wsi 
        and mask to specific size

        :param dim: dimensions
        :param factor: border increments
        :param threshold: min/max size
        :param operator: threshold limit
        :return new_dims: new border dimensions [(x1,y1),(x2,y2)]
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
    def get_border(self,space=100):
        """
        Generate border around max/min annotation points
        :param space: gap between max/min annotation point and border
        :self._border: border dimensions [(x1,y1),(x2,y2)]
        """
        if self.annotations is None:
            self._border=[[0,self.dims[0]],[0,self.dims[1]]]
        else:
            coordinates=self.annotations.annotations
            coordinates = list(chain(*list(coordinates.values())))
            coordinates=list(chain(*coordinates))
            f=lambda x: (min(x)-space, max(x)+space)
            self._border=list(map(f, list(zip(*coordinates))))
        mag_factor=Slide.MAG_fACTORS[self.mag]
        f=lambda x: (x[0]/mag_factor,x[1]/mag_factor)
        self._border=list(map(f,self._border))

        return self._border

    #Need to do min size in terms of micrometers not pixels
    def detect_components(self,level_dims=6,num_component=None,min_size=None):
        """
        Find the largest section on the slide
        :param down_factor: 
        :return image: image containing contour around detected section
        :return self._border: [(x1,x2),(y1,y2)] around detected section
        """
        new_dims=self.level_dimensions[6]
        image=np.array(self.get_thumbnail(self.level_dimensions[6]))
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blur=cv2.bilateralFilter(np.bitwise_not(gray),9,100,100)
        _,thresh=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if num_component is not None:
            idx=sorted([(cv2.contourArea(c),i) for i,c in enumerate(contours)])
            contours=[contours[i] for c, i in idx]
            contours=contours[-num_component:]
        if min_size is not None:
            contours=list(map(lambda x, y: cv2.contourArea(x),contours))
            contours=[c for c in contours if c>min_area]
        borders=[]
        components=[]
        image_new=image.copy()
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            x_scale=self.dims[0]/new_dims[0]
            y_scale=self.dims[1]/new_dims[1]
            x1=round(x_scale*x)
            x2=round(x_scale*(x+w))
            y1=round(y_scale*y)
            y2=round(y_scale*(y-h))
            self._border=[(x1,x2),(y1,y2)]
            image_new=cv2.rectangle(image_new,(x,y),(x+w,y+h),(0,255,0),2)
            components.append(image_new)
            borders.append([(x1,x2),(y1,y2)])            
        return components, borders 
    

    def generate_region(self, mag=0, x=None, y=None, x_size=None, y_size=None,
                        scale_border=False, factor=1, threshold=None, operator='=>'):
        """
        Extracts specific regions of the slide
        :param mag: magnification level
        :param x: min x coordinate
        :param y: min y coordinate
        :param x_size: x dim size 
        :param y_size: y dim size
        :param scale_border: resize border
        :param factor:
        :param threshold:
        :param operator:
        :return: extracted region (RGB ndarray)
        """
        if x is None:
            self.get_border()
            x, y = self._border        
        if x is not None:
            if isinstance(x,tuple):
                if x_size is None:
                    x_min, x_max=x
                    x_size=x_max-x_min
                elif x_size is not None:
                    x_min=x[0]
                    x_max=x_min+x_size
            elif isinstance(x,int):
                x_min=x
                x_max=x+x_size
        if y is not None:
            if isinstance(y,tuple):
                if y_size is None:
                    y_min, y_max=y
                    y_size=y_max-y_min
                elif y_size is not None:
                    y_min=y[0]
                    y_max=y_min+y_size
            elif isinstance(y,int):
                y_min=y
                y_max=y_min+y_size
        if scale_border:
            x_size = Slide.resize_border(x_size, factor, threshold, operator)
            y_size = Slide.resize_border(y_size, factor, threshold, operator)
        if (x_min+x_size)>self.dimensions[0]:
            x_size=self.dimensions[0]-x_min
        if (y_min+y_size)>self.dimensions[1]:
            y_size=self.dimensions[1]-y_min
        x_size=int(x_size/Slide.MAG_fACTORS[mag])
        y_size=int(y_size/Slide.MAG_fACTORS[mag])
        region=self.read_region((x_min,y_min),mag,(x_size, y_size))
        mask=self.generate_mask()[x_min:x_min+x_size,y_min:y_min+y_size]
        return np.array(region.convert('RGB')), mask

    def save(self, path, size=(2000,2000), mask=False):
        """
        Save thumbnail of slide in image file format
        :param path:
        :param size:
        :param mask:
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
    Returns dictionary of coordinates of ROIs. Reads annotation 
    files in either xml and json format and returns a dictionary 
    containing x,y coordinates for each region of interest in the 
    annotation

    :param path: string path to annotation file
    :param annotation_type: file type
    :param labels: list of ROI names ['roi1',roi2']
    :param _annotations: dictonary with return files
                      {roi1:[[x1,y1],[x2,y2],...[xn,yn],...roim:[]}
    """
    def __init__(self, path, source,labels=None, encode=False):
        self.paths=path if isinstance(path,list) else [path]
        self.source=source
        self.labels=labels
        self.encode=encode
        self._annotations=None
        self._generate_annotations()

    def __repr__(self):
        numbers=[len(v) for k, v in self._annotations.items()]
        df=pd.DataFrame({"classes":self.labels,"number":numbers})
        return str(df)

    @property
    def keys(self):
        return list(self.annotations.keys())

    @property
    def values(self):
        return list(self.annotations.values())

    @property
    def annotations(self):
        if self.encode:
            annotations=self.encode_keys()
            self.encode=False
        else:
            annotations=self._annotations
        return annotations

    @property
    def class_key(self):
        if self.labels is None:
            self.labels=list(self._annotations.keys())
        print(self.labels)
        class_key={l:i+1 for i, l in enumerate(self.labels)}
        return class_key

    @property
    def numbers(self):
        numbers=[len(v) for k, v in self._annotations.items()]
        return dict(zip(self.labels,numbers))


    def _generate_annotations(self):
        """
        Calls appropriate method for file type.
        return: annotations: dictionary of coordinates
        """
        self._annotations={}
        if not isinstance(self.paths,list):
            self._paths=[self.paths] 
        if self.source is not None:
            for p in self.paths:
                annotations=getattr(self,'_'+self.source)(p)
                for k, v in annotations.items():
                    if k in self._annotations:
                        self._annotations[k].append(v)
                    else:
                        self._annotations[k]=v
        if self.labels is not None:
            self._annotations=self.filter_labels(self.labels)
        else:
            self.labels=list(self._annotations.keys())
        

    def filter_labels(self, labels):
        """
        remove labels from annotations
        :param labels: label list to remove
        :return annotations: filtered annotation dictionary
        """
        self.labels=labels
        keys = list(self._annotations.keys())
        for k in keys:
            if k not in labels:
                del self._annotations[k]
        return self._annotations


    def rename_labels(self,names):
        """
        rename annotation labels
        :param names: dictionary {current_labels:new_labels}
        """
        for k,v in names.items():
            self._annotations[v]=self._annotations.pop(k)
        self.labels=list(self._annotations.keys())        
    

    def encode_keys(self):
        """
        encode labels as integer values
        """
        annotations={self.class_key[k]: v for k,v in self._annotations.items()}
        return annotations


    def _imagej(self,path):
        """
        Parses xml files
        :param path:
        :return annotations: dict of coordinates
        """
        tree=ET.parse(path)
        root=tree.getroot()
        anns=root.findall('Annotation')
        labels=list(root.iter('Annotation'))
        labels=list(set([i.attrib['Name'] for i in labels]))
        self.labels.extend(labels)
        annotations={l:[] for l in labels}
        for i in anns:
            label=i.attrib['Name']
            instances=list(i.iter('Vertices'))
            for j in instances:
                coordinates=list(j.iter('Vertex'))
                coordinates=[(c.attrib['X'],c.attrib['Y']) for c in coordinates]
                coordinates=[(round(float(c[0])),round(float(c[1]))) for c in coordinates]
                annotations[label]=annotations[label]+[coordinates]
        return annotations


    def _asap(self,path):
        """
        Parses _asap files
        :param path:
        :return annotations: dict of coordinates
        """
        tree=ET.parse(path)
        root=tree.getroot()
        ns=root[0].findall('Annotation')
        labels=list(root.iter('Annotation'))
        self.labels=list(set([i.attrib['PartOfGroup'] for i in labels]))
        annotations={l:[] for l in labels}
        for i in ns:
            coordinates=list(i.iter('Coordinate'))
            coordinates=[(float(c.attrib['X']),float(c.attrib['Y'])) for c in coordinates]
            coordinates=[(round(c[0]),round(c[1])) for c in coordinates]
            label=i.attrib['PartOfGroup']
            annotations[label]=annotations[label]+[coordinates]

        annotations = {self.class_key[k]: v for k,v in annotations.items()}
        return annotations

    
    def _qupath(self,path):
        """
        Parses qupath annotation json files
        :param path: json file path
        :return annotations: dictionary of annotations
        """
        annotations={}
        with open(path) as json_file:
            j=json.load(json_file)
        for a in j:
            c=a['properties']['classification']['name']
            geometry=a['geometry']['type']
            coordinates=a['geometry']['coordinates']
            if c not in annotations:
                annotations[c]=[]
            if geometry=="LineString":
                points=[[int(i[0]),int(i[1])] for i in coordinates]
            elif geometry=="Polygon":    
                for a2 in coordinates:
                    points=[[int(i[0]),int(i[1])] for i in a2]
            elif geometry=="MultiPolygon":
                for a2 in coordinates:
                    for a3 in a2:
                        points=[[int(i[0]),int(i[1])] for i in a3]
            annotations[c].append(points)
        return annotations


    def _json(self,path):
        """
        Parses json file with following structure.
        :param path:
        :return annotations: dict of coordinates
        """
        with open(path) as json_file:
            json_annotations=json.load(json_file)
        
        labels=list(json_annotations.keys())
        self.labels.extend(labels) 
        annotations = {k: [[[int(i['x']), int(i['y'])] for i in v2] 
                       for v2 in v.values()] for k, v in json_annotations.items()}
        return annotations


    def _dataframe(self):
        """
        Parses dataframe with following structure
        """ 
        anns_df=pd.read_csv(path)
        anns_df.fillna('undefined', inplace=True)
        anns_df.set_index('labels',drop=True,inplace=True)
        self.labels=list(set(anns_df.index))
        annotations={l: list(zip(anns_df.loc[l].x,anns_df.loc[l].y)) for l in
                     self.labels}

        annotations = {self.class_key[k]: v for k,v in annotations.items()}
        self._annotations=annotations


    def _csv(self,path):
        """
        Parses csv file with following structure
        :param path: 
        :return annotations: dict of coordinates
        """
        anns_df=pd.read_csv(path)
        anns_df.fillna('undefined', inplace=True)
        anns_df.set_index('labels',drop=True,inplace=True)
        self.labels=list(set(anns_df.index))
        annotations={l: list(zip(anns_df.loc[l].x,anns_df.loc[l].y)) for l in
                     self.labels}

        annotations = {self.class_key[k]: v for k,v in annotations.items()}
        self._annotations=annotations
        return annotations


    def df(self):
        """
        Returns dataframe of annotations.
        :return :dataframe of annotations
        """
        #key={v:k for k,v in self.class_key.items()}
        labels=[[l]*len(self._annotations[l]) for l in self._annotations.keys()]
        labels=chain(*labels)
        #labels=[key[l] for l in labels]
        x_values=[xi[0] for x in list(self._annotations.values()) for xi in x]
        y_values=[yi[1] for y in list(self._annotations.values()) for yi in y]
        df=pd.DataFrame({'labels':labels,'x':x_values,'y':y_values})

        return df


    def save(self,save_path):
        """
        Save down annotations in csv file.
        :param save_path: path to save annotations
        """
        self.df().to_csv(save_path)
