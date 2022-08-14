import os
import json
import glob
import argparse
import math

import cv2
import numpy as np
import tensorflow as tf



class TFRecordRead(self):
    def __init__(self):
        self.db_path=db_path
        self.dims=dims


    @property
    def num(self):
        pass


    @static_method
    def _print_progress(self,i):
        complete = float(count)/total
        print('\r- Progress: {0:.1%}'.format(complete), flush=True)

    
    @static_method
    def _wrap_int64(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    @static_method
    def _wrap_bytes(self,value):
    '''
    convert value to bytes
    param: value: image
    returns: 
    '''
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def convert(self, patch):
        writer=tf.io.TFRecordWriter(self.db_path)
        for i, (p, image) in enumerate(patch.extract_patches()):
            self._print_progress(i)
            image = tf.image.encode_png(image)
            
            data = {
                'image': wrap_bytes(image),
                'name': wrap_bytes(p['name']),
                'dims': wrap_int64(dims[0]) 
                }
               
            features = tf.train.Features(feature=data)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)

