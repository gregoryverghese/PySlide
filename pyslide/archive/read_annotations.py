#!/usr/local/share python
# -*- coding: utf-8 -*-

'''
read_annotations.py
'''

import os
import json
import glob
import argparse

import cv2
import openslide
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as import Path
from itertools import chain

__author__ = 'Gregory Verghese'
__email__ = 'gregory.verghese@kcl.ac.uk'


class ImageScopeAnnotations():
    def __init__(self):
        self.xml_path = xml_path
        pass


    def readxml(self, xmlFileName):
        
        tree = ET.parse(xmlFileName)
        root = tree.getroot()
        pixelSpacing = float(root.get('MicronsPerPixel'))

        regions = {n.attrib['Name']: n[1].findall('Region') for n in root}
        labelAreas = {}

        for a in regions:
            region = {}
            for r in regions[a]:
                iD = int(r.get('Id'))
                area = r.attrib['AreaMicrons']
                length = r.attrib['LengthMicrons']
                vertices = r[1].findall('Vertex')
                f = lambda x: (int(float(x.attrib['X'])), int(float(x.attrib['Y'])))
                coords = list(map(f, vertices))
                region[iD] = dict(zip(('area', 'length', 'coords'), (area, length, coords)))

            labelAreas[a] = region

        return labelAreas


    def get_annotations(self, ndpi, xmlFile):
        
        if not os.path.exists(xmlFile):
            print('no ImageJ xml file')
            return None, None

        print('Getting annotations from ImageJ xml')
        xmlAnnotations = self.getRegions(xmlFile)

        border = xmlAnnotations[''][1]['coords']
        boundaries = list(map(lambda x: (min(x), max(x)), list(zip(*border))))

        keys = xmlAnnotations.keys()
        for k in list(keys):
            if k not in self.classKey:
                del xmlAnnotations[k]

        if not xmlAnnotations:
            print('No {} annotations in ImageJ'.format(self.feature))
            return None, None

        values = sum([len(xmlAnnotations[k]) for k in keys])
        if values==0:
            print('No coordinates for {} annotations in ImageJ'.format(self.feature))
            return None, None

        print('ImageJ annotations exist for {}'.format(self.feature))
        annotations = {self.classKey[k]: [v2['coords'] for k2, v2 in v.items()] for k, v in xmlAnnotations.items()}
        return annotations, boundaries
