#!/usr/share/env python
# -*- coding: utf-8 -*-

'''
save.py: provides functions for saving patches down in various formats
'''

import os
import glob
import argparse

import cv2
import PIL
import numpy
import pickle

import openslide

__author__ = ' Gregory Verghese'
__email__  = 'gregory.verghese@kcl.ac.uk'


class Store():
    def __init__(self, patches, masks=None):
        self.patches = patches
        self.masks = 


    def _get_patches(self):
        return self.patches.extract_patches()

       
    def hd5(self):

        patches = _get_patches()

        f = h5py.File(os.path.join(self.out_path, name, '.h5'), 'w')

        patch_ds = f.create_dataset(
                'name', np.shape(patches), h5py.h5t.STD_U8BE, data=patches
        )
         
        if mask not None:
            mask_ds = f.create_dateset(
                'mask', np.shape(patches), h5py.h5t.STD_U8BE, data=masks 
        (
        
        if labels not None:
            meta_set = f.create_dataset(
                'meta', np.shape(labels), h5py.h5t.STD_U8BE, data=labels
        )

        f.close()

        pass


    def images(self, extension):
        
        patches = _get_patches()
        if masks:
            masks = _get_masks():

        for p in patches:
            patch = Image.fromarray(p)
            patch.save(os.path.join(self.patch_path, label + extension))

        if masks:
            for m in masks:
                mask = Image.fromarray(m)
                patch.save(os.path.join(self.mask_path, label + extension))

        
        if meta:
           pass



    def _Tfrecords():






class _Tfrecords():


    def getShardNumber(images, masks, shardSize=0.1, unit=10**9):

    maskMem = sum(os.path.getsize(f) for f in masks if os.path.isfile(f))
    imageMem = sum(os.path.getsize(f) for f in images if os.path.isfile(f))
    totalMem = (maskMem+imageMem)/unit
    print('Image memory: {}, Mask memory: {}, Total memory: {}'.format(imageMem, maskMem, totalMem))

    shardNum = int(np.ceil(totalMem/shardSize))
    imgPerShard = int(np.floor(len(images)/shardNum))

    return shardNum, imgPerShard


def printProgress(count, total):

    complete = float(count)/total
    print('\r- Progress: {0:.1%}'.format(complete), flush=True)


def wrapInt64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrapFloat(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def wrapBytes(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(imageFiles, maskFiles, tfRecordPath, dim=None):

    numImgs = len(imageFiles)
    check=[]
    with tf.io.TFRecordWriter(tfRecordPath) as writer:
        for i, (img, m) in enumerate(zip(imageFiles, maskFiles)):
            printProgress(i,numImgs)
            imgName = os.path.basename(img)[:-4]
            maskName = os.path.basename(m)[:-10]
            mPath = os.path.dirname(m)

            m = os.path.join(mPath, os.path.basename(img[:-4]) + '_masks.png')
            maskName = os.path.basename(m)
            if not os.path.exists(m):
                check.append(maskName)
                continue

            maskName = os.path.basename(m)

            image = tf.keras.preprocessing.image.load_img(img)
            image = tf.keras.preprocessing.image.img_to_array(image, dtype=np.uint8)
            dims = image.shape
            image = tf.image.encode_png(image)

            mask = tf.keras.preprocessing.image.load_img(m)
            mask = tf.keras.preprocessing.image.img_to_array(mask, dtype=np.uint8)
            mask = tf.image.encode_png(mask)

            data = {
                'image': wrapBytes(image),
                'mask': wrapBytes(mask),
                'imageName': wrapBytes(os.path.basename(img)[:-4].encode('utf-8')),
                'maskName': wrapBytes(os.path.basename(m)[:-4].encode('utf-8')),
                'dims': wrapInt64(dims[0])
                }

            features = tf.train.Features(feature=data)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)

        print('Number of errors: {}'.format(len(check)))


def doConversion(imgs, masks, shardNum, num, outPath, outDir):

    print('shardNum', shardNum)
    print('num', num)


    for i in range(0, shardNum):
        shardImgs = imgs[i*num:num*(i+1)]
        shardMasks = masks[i*num:num*(i+1)]
        convert(shardImgs, shardMasks, os.path.join(outPath,outDir,str(i)+'.tfrecords'), dim=None)

    if shardNum > 1:
        shardImgs = imgs[i*num:]
        shardMasks = masks[i*num:]

    convert(shardImgs, shardMasks, os.path.join(outPath,outDir,str(i)+'.tfrecords'), dim=None)

