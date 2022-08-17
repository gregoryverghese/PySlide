import os
import sys
import json
import glob
import argparse
import math

import cv2
import numpy as np
import tensorflow as tf



class TFRecordWrite(self):
    def __init__(self,
                 db_path,
                 dims,
                 shard_size=0.01,
                 unit=10**9):

        self.db_path=db_path
        self.dims=dims
        self.shard_size=0.01 
        self.unit=10**9

    
    @static_method
    def _print_progress(self,i):
        complete = float(count)/total
        print('\r- Progress: {0:.1%}'.format(complete), flush=True)

    
    @static_method
    def _wrap_int64(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    @static_method
    def _wrap_bytes(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    
    @property
    def mem_size(self):
        return mem = sum(sys.getsizeof(i.tobytes()) 
                        for i in self.patch.extract_patches())

    @property
    def shard_number(self):
        return int(np.ceil(mem/self.shard_size))


    @property
    def img_num_per_shard(self):
        return int(np.floor(len(self.patches_patches)/self.shard_size))


    def convert(self, patch):

        convert(shardImgs, shardMasks, os.path.join(outPath,outDir,str(i)+'.tfrecords'), dim=None
      
        writer=tf.io.TFRecordWriter(self.db_path)
        for i, (p, image) in enumerate(self.patch.extract_patches()):
            
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

