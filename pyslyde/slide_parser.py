import os
import glob
import json
import random
from typing import List, Tuple, Optional, Generator, Callable

import numpy as np
import cv2
import seaborn as sns
from matplotlib.path import Path
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
import pandas as pd
from itertools import chain
import operator as op

#from tiler.utilities import mask2rgb, reinhard
#from exceptions import StitchingMissingPatches
from pyslyde.io.lmdb_io import LMDBWrite
from pyslyde.io.disk_io import DiskWrite
from pyslyde.encoders.feature_extractor import FeatureGenerator
#from pyslide.io.tfrecords_io import TFRecordWrite


class WSIParser:
    def __init__(
            self,
            slide: OpenSlide,
            tile_dim: int,
            border: List[Tuple[int, int]],
            mag_level: int = 0,
            stain_normalizer = None
    ) -> None:
        super().__init__()
        self.slide = slide
        self.mag_level = mag_level
        self.tile_dims = (tile_dim, tile_dim)
        self.border = border

        self._x_min = int(self.border[0][1])
        self._x_max = int(self.border[1][1])
        self._y_min = int(self.border[0][0])
        self._y_max = int(self.border[1][0])
        self._downsample = int(slide.level_downsamples[mag_level])
        print('downsample', self._downsample)
        self._x_dim = int(tile_dim * self._downsample)
        self._y_dim = int(tile_dim * self._downsample)
        self._tiles: List[Tuple[int, int]] = []
        self._features: List[np.ndarray] = []
        self._number = len(self._tiles)

        self.stain_normalizer = stain_normalizer

    @property
    def number(self) -> int:
        """returns int"""
        return len(self._tiles)

    @property
    def tiles(self) -> List[Tuple[int, int]]:
        """returns list of tuples with (x,y) x and y are ints"""
        return self._tiles

    @tiles.setter
    def tiles(self, value: List[Tuple[int, int]]) -> None:
        """param list with tuples (x,y) int coordinates"""
        self._tiles = value

    @property
    def features(self) -> List[np.ndarray]:
        """returns list of numpy array"""
        return self._features

    @property
    def config(self) -> dict:
        """dictionary"""
        config = {
            'name': self.slide.name,
            'mag': self.mag_level,
            'size': self.size,
            'border': self.border,
            'number': self._number
        }
        return config

    def __repr__(self) -> str:
        """
        Object representation

        return: str(self.config)
        """
        return str(self.config)

    def _remove_edge_case(self, x: int, y: int) -> bool:
        """
        Remove edge cases based on dimensions of patch

        param: x: base x coordinate to test 
        param: y: base y coordinate to test
        return: boolean remove patch or not
        """
        remove = False
        if x + self._x_dim > self._x_max:
            remove = True
        if y + self._y_dim > self._y_max:
            remove = True
        return remove

    def _tile_downsample(self, image: np.ndarray, ds: int) -> np.ndarray:
        """
        param: image numpy array
        param: ds int
        return: image numpy array
        """
        if ds:
            x, y, _ = image.shape
            image = cv2.resize(image, (int(y / ds), int(x / ds)))
            print(f"Downsampled to shape {image.shape}")
        return image

    def tiler(
            self, 
            stride: Optional[int] = None, 
            edge_cases: bool = False
    ) -> int:
        """
        Generate tile coordinates based on border
        mag_level, and stride.

        param: stride: int: step size
        return: len(self._tiles): int Number of patches
        """
        stride = self.tile_dims[0] if stride is None else stride
        stride = stride * self._downsample
        print('stride', stride)
        self._tiles = []
        for x in range(self._x_min, self._x_max, stride):
            for y in range(self._y_min, self._y_max, stride):
                # if self._remove_edge_case(x, y):
                    # continue
                self._tiles.append((x, y))

        self._number = len(self._tiles)
        return self._number

    def extract_features(
            self,
            model_name: str,
            model_path: str,
            device: Optional[str] = None,
            downsample: Optional[int] = None,
            normalize: bool = False
    ) -> Generator[Tuple[Tuple[int, int], np.ndarray], None, None]:
        """
        param: model_name str
        param: model_path str
        param: device str
        param: downsample int
        param: normalize boolean
        yield: t numpy array
        yield: feature_vec numpy array
        """
        encode = FeatureGenerator(model_name, model_path)  # model_path is the path to the model's weights
        print(f'Extracting features...')
        print(f'checking again... {len(self.tiles)}')

        for i, t in enumerate(self.tiles):
            tile = self.extract_tile(t[0], t[1])
            if downsample:
                tile = self._tile_downsample(tile, downsample)
            if normalize and self.stain_normalizer is not None:
                self.stain_normalizer.normalize(tile)

            feature_vec = encode.forward_pass(tile)
            feature_vec = feature_vec.detach().cpu().numpy()
            print(f'{i}')
            yield t, feature_vec

    def filter_tissue(
            self,
            slide_mask: np.ndarray,
            label: int,
            threshold: float = 0.5
    ) -> int:
        """
        param: slide_mask numpy array
        param: label int
        param: threshold float
        return: int
        """
        slide_mask[slide_mask != label] = 0
        slide_mask[slide_mask == label] = 1
        tiles = self._tiles.copy()
        for t in self._tiles:
            x, y = (t[0], t[1])
            t_mask = slide_mask[x:x + self._x_dim, y:y + self._y_dim]
            if np.sum(t_mask) < threshold * (self._x_dim * self._y_dim):
                tiles.remove(t)

        self._tiles = tiles
        return len(self._tiles)

    def filter_tiles(self, filter_func: Callable[[np.ndarray], bool]) -> None:
        """
        Filter tiles using filtering function

        param: filter_func python function
        """
        tiles = self._tiles.copy()

        for i, tile in enumerate(self.extract_tiles()):
            if filter_func(tile):
                tiles.remove(tile)

        print(f'Removed {self.number - len(tiles)} tiles')
        self._tiles = tiles.copy()

    def sample_tiles(self, n: int) -> None:
        """
        param: n int
        """
        n = len(self._tiles) if n > len(self._tiles) else n
        sple_tiles = random.sample(self._tiles, n)
        self._tiles = sple_tiles

    def extract_tile(
            self,
            x: Optional[int] = None,
            y: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract individual patch from WSI.

        param: x int x coordinate
        param: y int y coordinate
        return: patch ndarray patch
        """
        tile = self.slide.read_region(
            (y, x),
            self.mag_level, 
            self.tile_dims
        )
        tile = np.array(tile.convert('RGB'))
        # print(tile.shape)
        return tile

    def extract_tiles(
            self,
            normalize: bool = False
    ) -> Generator[Tuple[Tuple[int, int], np.ndarray], None, None]:
        """
        Generator to extract all tiles

        yield: tile ndarray tile
        yield: p tile dict metadata
        """
        print('this is the final tile number', len(self._tiles))
        for t in self._tiles:
            tile = self.extract_tile(t[0], t[1])
            if normalize and self.stain_normalizer is not None:
                self.stain_normalizer.normalize(tile)
            yield t, tile

    @staticmethod
    def _save_to_disk(
            image: np.ndarray,
            path: str,
            x: Optional[int] = None,
            y: Optional[int] = None
    ) -> bool:
        """
        Save tile only if WSI x and y position
        known.

        param: image ndarray tile
        param: path str path to save
        param: x int x coordinate for filename
        param: y int y coordinate for filename
        return: bool
        """
        assert isinstance(y, int) and isinstance(x, int)
        filename = '_' + str(y) + '_' + str(x)
        image_path = path + filename + '.png'
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        status = cv2.imwrite(image_path, image)
        # print(status, image_path)
        return status
    
    def save(
        self,
        func: Generator[Tuple[Tuple[int, int], np.ndarray], None, None],
        tile_path: str,
        label_dir: bool = False,
        label_csv: bool = False,
        normalize: bool = False
    ) -> None:
        """
        Save the extracted tiles to disk.

        param: func Generator: A generator function that yields (coordinates, tile) tuples.
        param: tile_path str: The base directory where tiles will be saved. 
            Before being supplied to this method, it is derived as args.save_path/dir_/args.name
            by the caller script, where dir_ is 'tiles' if args.parser=='tiler' else 'features',
            and args.name is name of the WSI file excluding its file extension.
        param: label_dir bool: If True, saves tiles in subdirectories based on their label.
        param: label_csv bool: If True, saves tile metadata in a CSV file.
        param: normalize bool: If True, applies normalization to the tiles before saving.
        """
        os.makedirs(tile_path, exist_ok=True)

        metadata = []

        for (x, y), tile in func:
            if normalize and self.stain_normalizer is not None:
                self.stain_normalizer.normalize(tile)
            
            # Generate directory path
            save_dir = tile_path
            if label_dir:
                """tile_path is used as a named (i.e., labeled) directory.
                Without this, all tiles from separate WSIs will be saved in the same directory, but prefixed
                by their WSI name as the only differentiation."""
                save_dir = os.path.join(tile_path, os.path.basename(tile_path))
                os.makedirs(save_dir, exist_ok=True)

            # Save the tile image
            self._save_to_disk(tile, save_dir, x, y)
            
            # Optionally save metadata
            if label_csv:
                metadata.append({"y": y, "x": x, "path": save_dir + f"_{y}_{x}.png"})

        if label_csv:
            df = pd.DataFrame(metadata)
            df.to_csv(tile_path + "_metadata.csv", index=False)


    def to_lmdb(
        self,
        func: Generator[Tuple[Tuple[int, int], np.ndarray], None, None],
        db_path: str, 
        map_size: int,
        write_frequency: int = 10
    ) -> None:
        """
        Save to lmdb database.

        param: func Generator: A generator function that yields (coordinates, tile) tuples.
        param: db_path str: The base directory where tiles or features will be saved. 
        param: map_size int: map_size for lmdb
        param: write_frequency int: Controls batch commit of a transaction.
        """
        os.makedirs(db_path, exist_ok=True)
        lmdb_writer = LMDBWrite(db_path, map_size, write_frequency)
        lmdb_writer.write(func)


    def to_rocksdb(
        self,
        func: Generator[Tuple[Tuple[int, int], np.ndarray], None, None],
        db_path: str,
        write_frequency: int = 10
    ) -> None:
        """
        Save to rocksdb database.

        param: func Generator: A generator function that yields (coordinates, tile) tuples.
        param: db_path str: The base directory where tiles or features will be saved.
        param: write_frequency int: Controls batch commit of a transaction.
        """
        os.makedirs(db_path, exist_ok=True)
        writer = RocksDBWrite(db_path, write_frequency)
        writer.write(func)


    def feat_to_disk(
        self,
        func: Generator[Tuple[Tuple[int, int], np.ndarray], None, None],
        path: str,
        write_frequency: int = 10
    ) -> None:
        """
        Save to disk.

        param: func Generator: A generator function that yields (coordinates, tile) tuples.
        param: db_path str: The base directory where tiles or features will be saved.
        param: write_frequency int: Controls batch commit of a transaction.
        """
        os.makedirs(path, exist_ok=True)
        writer = DiskWrite(path, write_frequency)
        writer.write(func)  


class Stitching():

    MAG_FACTORS={0:1,1:2,2:4,3:8,4:16,5:32,5:64}

    def __init__(self,patch_path,
                 slide=None,
                 patching=None,
                 name=None,
                 step=None,
                 border=None,
                 mag_level=0):

        self.patch_path=patch_path
        patch_files=glob.glob(os.path.join(self.patch_path,'*'))
        self.patch_files=[os.path.basename(p) for p in patch_files]
        print(patch_files[0])
        self.fext=self.patch_files[0].split('.')[-1]
        self.slide=slide
        self.coords=self._get_coords()
        self.mag_level=mag_level
        print(self.patch_files)

        if name is not None:
            self.name=name
        elif patching is not None:
            self.name=self.patching.slide.name
        else:
            raise TypeError("missing name")

        if patching is not None:
            self.border=patching.slide.border
        else:
            self.border=self._get_border()
 
        if patching is not None:
            self.mag_level=patching.mag_level
        elif mag_level is not None:
            self.mag_level=mag_level

        self._completeness()
        print(self.config)
        

    @property
    def config(self):
        config={'name':self.name,
                'mag':self.mag_level,
                'step':self.step,
                'border':self.border,
                'patches':len(self.patch_files)}
        return config


    def __repr__(self):
        return str(self.config)


    @property
    def step(self):
        self._step=self._get_step()
        return self._step


    @property
    def mag_factor(self):
         return Stitching.MAG_FACTORS[self.mag_level]
    
    #TODO: check required coordinates according to parameters
    def _get_coords(self):
        """
        return coordinates of patches based on patch filesnames
        :return self._coords: list [(x1,y1),(x2,y2), ..., (xn,yn)]
        """
        patch_files=glob.glob(os.path.join(self.patch_path,'*'))
        coords=[(int(f.split('_')[-2:][0]),int(f.split('_')[-2:][1][:-4]))
                for f in patch_files]

        self._coords=coords
        return self._coords


    def _get_border(self):
        """
        calculate border based on coordinate maxima and minima
        :return [[xmin,xmax],[ymin,ymax]]
        """
        coords=self._get_coords()
        xmax=max([c[0] for c in coords])
        xmin=min([c[0] for c in coords])
        ymax=max([c[1] for c in coords])
        ymin=min([c[1] for c in coords])

        return [[xmin,xmax],[ymin,ymax]]


    def _get_step(self):
        """
        calculate step based on patch filenames
        :return int(step/self.mag_factor)
        """
        coords=self._get_coords()
        xs=[c[0] for c in coords]
        step=min([abs(x1-x2) for x1, x2 in zip(xs, xs[1:]) if abs(x1-x2)!=0])
        return int(step/self.mag_factor)


    def _completeness(self):
        """
        check patch set is complete to stitch entire image
        based on the coordinates. Raises MissingPatches error
        if missing
        """
        missing_patches=[]
        for (p_name,_,_) in self._patches():
            print(p_name)
            if p_name not in self.patch_files:
                missing_patches.append(p_name)
        if len(missing_patches)>0:        
            raise StitchingMissingPatches(missing_patches)


    def _patches(self):
        """
        return patches metadata (name,(x,y))
        """
        step=self.step*self.mag_factor
        xmin,xmax=self.border[0][0],self.border[0][1]
        ymin,ymax=self.border[1][0],self.border[1][1]
        for i,x in enumerate(range(xmin,xmax+step,step)):
            for j,y in enumerate(range(ymin,ymax+step,step)):
                filename=self.name+'_'+str(x)+'_'+str(y)+'.'+self.fext
                yield filename,x,y



    def stitch(self,size=None):
        """
        stitches patches together to create entire
        slide representation. Size argument 
        determnines image size
        :param size: (x_size,y_size)
        """
        xmin,xmax=self.border[0][0],self.border[0][1]
        ymin,ymax=self.border[1][0],self.border[1][1]
        z=self.step*self.mag_factor
        xnew=(xmax+z-xmin)/self.mag_factor
        ynew=(ymax+z-ymin)/self.mag_factor
        canvas=np.zeros((int(ynew),int(xnew),3))
        
        #what do we do if it is none
        if size is not None:
            x_num=(xmax-xmin)/(self.step*self.mag_factor)+1
            y_num=(ymax-ymin)/(self.step*self.mag_factor)+1
            xdim_new=(int((size[0]/x_num))+1)*x_num
            ydim_new=(int((size[1]/y_num))+1)*y_num
            p_xsize=int(xdim_new/x_num)
            p_ysize=int(ydim_new/y_num)
            
        canvas=np.zeros((int(ydim_new),int(xdim_new),3))
        for filename,x,y in self._patches():
            p=cv2.imread(os.path.join(self.patch_path,filename))
            if size is not None:
                p=cv2.resize(p,(p_xsize,p_ysize))
                x=int(((x-xmin)/(self.step*self.mag_factor))*p_xsize)
                y=int(((y-ymin)/(self.step*self.mag_factor))*p_ysize)
            canvas[y:y+p_ysize,x:x+p_xsize,:]=p
        return canvas.astype(np.uint8)


