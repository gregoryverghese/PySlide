import os
import glob
import pickle
from PIL import Image

import lmdb
import numpy as np
from torch.utils.data import DataLoader, Dataset


class LMDBWrite():
    def __init__(self,db_path,map_size,write_frequency=10):
        self.db_path=db_path
        self.map_size=map_size
        self.env=lmdb.open(self.db_path, 
                           map_size=5e9,
                           writemap=True)
        self.write_frequency=write_frequency

    def __repr__(self):
        return f'LMBDWrite(size: {self.map_size}, path: {self.db_path})'
    

    def _serialize(self,image):
        image_bytes = image.tobytes()
        return image_bytes


    def _print_progress(self,i,total):
        complete = float(i)/total
        print(f'\r- Progress: {complete:.1%}', end='\r')


    def write(self,patch): 
        txn=self.env.begin(write=True)
        for i, (image, p) in enumerate(patch.extract_patches()):
            value=self._serialize(image)
            key = f"{p['name']}"
            txn.put(key.encode("ascii"), pickle.dumps(value))
            self._print_progress(i,len(patch._patches))
            if i % self.write_frequency == 0:
                txn.commit()
                txn = self.env.begin(write=True)
        txn.commit()
        self.env.close()


    def close(self):
        self.env.close()


class LMDBRead():
    def __init__(self, db_path, image_size):
        self.db_path=db_path
        self.env=lmdb.open(self.db_path,
                           readonly=True,
                           lock=False
                           )
        self.image_size=image_size


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
        image = np.frombuffer(image, dtype=np.uint8)
        image = image.reshape(self.image_size)
        return image

