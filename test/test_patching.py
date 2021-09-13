import os
import json
import xml.etree.ElementTree as ET

import unittest
import numpy as np
from itertools import chain

from pyslide.patching import Patching, Slide


class TestPatching(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path='14.90610 C L2.11.ndpi'


    def setUp(self):
        self.wsi=Slide(self.path)
        _,_=self.wsi.detect_component()


    def test_generate_patches(self):
        p_obj_1=Patching(self.wsi)
        num=p_obj_1.generate_patches(256)
        self.assertEqual(num,2703)
        
        num=p_obj_1.generate_patches(1024)
        self.assertEqual(num,156)
        
        num=p_obj_1.generate_patches(256,mode='focus')



        p_obj_2=Patching(self.wsi,mag_level=4)
        num=p_obj_2.generate_patches(256)
        self.assertEqual(num,4)



    
    

if __name__=='__main__':
   unittest.main() 
