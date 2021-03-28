import os
import json
import xml.etree.ElementTree as ET

import unittest
import numpy as np
from itertools import chain

from pyslide.slide import Annotations, Slide


class TestSlide(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.ndpi_path='14.90610 C L2.11.ndpi'
        cls.json_path='14.90610 C L2.11.json'

    def setUp(self):
        ann_obj=Annotations(self.json_path)
        self.annotations=ann_obj.generate_annotations()
        self.slide_obj=Slide(self.ndpi_path,annotations=self.annotations)
        

    def test_slide_mask(self):
        mask=self.slide_obj.slide_mask((2000,2000))
        self.assertEqual(mask.shape,(2000,2000,3))
        labels=np.unique(mask)
        self.assertEqual(len(labels),2)
        self.assertEqual(set(labels),{0,255})

    
    def test_generate_annotations(self):
        annotations=self.slide_obj.generate_annotations(self.json_path)
        self.assertEqual(len(annotations),3)
        self.assertEqual(set(annotations.keys()),{0,1,2})

        num_anns={len(annotations[k]) for k in annotations.keys()}
        self.assertEqual(num_anns,{13, 21, 14})


    def test_resize_border(self):
        
        _,border=self.slide_obj.detect_component()
        new_dims=[self.slide_obj.resize_border(b,512) for b in chain(*self.slide_obj.border)]
        #why in jupyter notebook 7168
        self.assertEqual(new_dims,[7680, 21504, 16384, 29184])
    

    def test_get_border(self):
        border1=self.slide_obj.get_border()
        self.assertEqual(border1,[(7689, 20172), (16026, 29254)])
        border2=self.slide_obj.get_border(space=200)
        self.assertEqual(border2,[(7589, 20272), (15926, 29354)])
        

    def test_detect_components(self):
        image,border=self.slide_obj.detect_component() 
        #Check why 519 in jupyter notebook
        self.assertEqual(image.shape,(520, 983, 3))
        #Check why 7100 in jupyter notebook
        self.assertEqual(border,[(7200, 21001), (15890, 29182)])


    def test_generate_region(self):

        region,mask=self.slide_obj.generate_region()
        self.assertEqual(region.shape,(13228, 12483, 3))
        self.assertEqual(list(np.unique(mask)),[0,255])
        

'''
class Annotations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.imagej_path='data/annotations/imagej_annotations.xml'
        cls.asap_path='data/annotations/asap_annotations.xml'
        cls.json_path='data/annotations/json_annotations.json'
        cls.csv_path='data/annotations/csv_annotations.csv'
        cls.ann_obj=Annotations


    def test_imageJ(self):
        annotations=self.ann_obj(self.imagej_path)._imagej()
        self.assertEqual(list(annotations.keys()),[0,1,2,3,4])
        self.assertTrue(len(annotations),5)

        sizes={len(annotations[a]) for a in annotations}
        self.assertEqual(sizes,{11538, 4, 6721, 2002, 7129})
        

    def test_asap(self):
        annotations=self.ann_obj(self.asap_path)._asap()
        self.assertEqual(list(annotations.keys()),[0,1])
        self.assertEqual(len(annotations),2)

        sizes={len(annotations[a]) for a in annotations}
        self.assertEqual(sizes,{2, 327})


    def test_json(self):
        annotations=self.ann_obj(self.json_path)._json()
        self.assertEqual(list(annotations.keys()),[0,1,2])
        self.assertEqual(len(annotations),3)

        sizes={len(annotations[a]) for a in annotations}
        self.assertEqual(sizes,{387, 243, 515})
        

    def test_csv(self):

        annotations=self.ann_obj(self.csv_path)._csv()
        self.assertEqual(list(annotations.keys()),[0,1,2,3,4])
        self.assertEqual(len(annotations),5)

        sizes={len(annotations[a]) for a in annotations}
        self.assertEqual(sizes,{11538, 4, 6721, 2002, 7129})

'''

if __name__=='__main__':
    unittest.main()



