#!/usr/local/share python
# -*- coding: utf-8 -*-

'''
def resizeImage(dim, factor=2048, threshold=0, op=operator.gt):
    boundaries = [factor*i for i in range(100000)]
    boundaries = [f for f in boundaries if op(f,threshold)]
    diff = list(map(lambda x: abs(dim-x), boundaries))
    newDim = boundaries[diff.index(min(diff))]

    return newDimutilities.py
'''

import numpy as np
import xml.etree.ElementTree as ET
import seaborn as sns
from itertools import chain


def mask2rgb(mask):
    n_classes=len(np.unique(mask))
    colors=sns.color_palette('hls',n_classes)
    rgb_mask=np.zeros(mask.shape+(3,))
    for c in range(1,n_classes+1):
        t=(mask==c)
        rgb_mask[:,:,0][t]=colors[c-1][0]
        rgb_mask[:,:,1][t]=colors[c-1][1]
        rgb_mask[:,:,2][t]=colors[c-1][2]
    return rgb_mask


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


def oneHotToMask(onehot):
    nClasses =  onehot.shape[-1]
    idx = tf.argmax(onehot, axis=-1)
    colors = sns.color_palette('hls', nClasses)
    multimask = tf.gather(colors, idx)
    multimask = np.where(multimask[:,:,:]==colors[0], 0, multimask[:,:,:])

    return multimask
