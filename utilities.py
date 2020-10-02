#!/usr/local/share python
# -*- coding: utf-8 -*-

'''
utilities.py
'''

import numpy as np
import xml.etree.ElementTree as ET
from itertools import chain

def draw_boundary(annotations, offset=100):

    annotations = list(chain(*[annotations[f] for f in annotations]))
    coords = list(chain(*annotations))
    boundaries = list(map(lambda x: (min(x)-offset, max(x)+offset), list(zip(*coords))))
   
    return boundaries


def getRegions(xmlFileName):

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
