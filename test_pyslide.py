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
        cls.ann_obj=Annotations('data/annotations/imagej_annotations.xml')


    def test__imageJ(self):
        annotations=self.ann_obj._imageJ()
        self.assertTrue(len(annotations),5)
        

    def test__asap(self):
        pass


    def test_json(self):
        pass


    def test__csv(self):
        pass


if __name__=='__main__':
    unittest.main()


