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
        cls.json_path='data/annotations/json_path.json'
        cls.ann_obj=Annotations


    def test_imageJ(self):
        annotations=self.ann_obj(self.imagej_path)._imageJ()
        self.assertEqual(list(annotations.keys()),[0,1,2,3,4])
        self.assertTrue(len(annotations),5)
        

    def test_asap(self):
        pass


    def test_json(self):
        pass


    def test__csv(self):
        pass


if __name__=='__main__':
    unittest.main()



