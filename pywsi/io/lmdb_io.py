import os
import glob
import pickle
from PIL import Image

import lmdb
import numpy as np
from torch.utils.data import DataLoader, Dataset


class NpyObject():
    def __init__(self, ndarray):
        self.ndarray = ndarray.tobytes()
        #self.label = label
        self.size = ndarray.shape
        self.dtype = ndarray.dtype

    def get_ndarray(self):
        ndarray = np.frombuffer(self.ndarray, dtype=self.dtype)
        return ndarray.reshape(self.size)


class LMDBWrite():
    def __init__(self,db_path,map_size,write_frequency=10):
        self.db_path = db_path
        self.map_size = map_size
        print(f"db path and map_size: {self.db_path} {self.map_size}")
        self.env = lmdb.open(path=self.db_path,map_size=self.map_size,writemap=True)
        self.write_frequency = write_frequency


    def __repr__(self):
        return f'LMBDWrite(size: {self.map_size}, path: {self.db_path})'
    

    def _print_progress(self,i,total):
        complete = float(i)/total
        print(f'\r- Progress: {complete:.1%}', end='\r')


    def write(self,parser):
        print("Beginning writing to db ...")
        txn=self.env.begin(write=True)
        for i, (p, tile) in enumerate(parser):
            name = str(p[1])+'_'+str(p[0])
            key = f"{name}"
            value=NpyObject(tile)
            txn.put(key.encode("ascii"), pickle.dumps(value))
            #self._print_progress(i,len(patch._patches))
            if i % self.write_frequency == 0:
                txn.commit()
                txn = self.env.begin(write=True)
        txn.commit()
        self.env.close()
        print("Writing to db done")


    def write_image(self,image,name): 
        txn=self.env.begin(write=True)
        value=NpyObject(image)
        b=pickle.dumps(value)
        key = f"{name}"
        txn.put(key.encode('ascii'), b)
        txn.commit()


    def close(self):
        self.env.close()


class LMDBRead():
    def __init__(self, db_path, image_size=None):
        self.db_path=db_path
        self.env=lmdb.open(self.db_path,
                           readonly=True,
                           lock=False
                           )
        #self.image_size=image_size


    @property
    def num_keys(self):
        with self.env.begin() as txn:
            length = txn.stat()['entries']
        return length


    def __repr__(self):
        return f'LMDBRead(path: {self.db_path})'


    def get_keys(self):
        txn = self.env.begin()
        keys = [k for k, _ in txn.cursor()]
        #self.env.close()
        return keys


    def read_image(self,key):
        txn = self.env.begin()
        data = txn.get(key)
        image = pickle.loads(data)
        image=image.get_ndarray()
        #image = np.frombuffer(image, dtype=np.uint8)
        #image = image.reshape(self.image_size)
        return image

