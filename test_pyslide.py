import os
import json
import xml.etree.ElementTree as ET

import unittest
import numpy as np

from pyslide.slide import Annotations


class TestSlide():
    pass


class TestAnnotations(unittest.TestCase):

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


if __name__=='__main__':
    unittest.main()



