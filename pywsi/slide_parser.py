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
from tiler.lmdb_io import LMDBWrite
from tiler.disk_io import DiskWrite
from tiler.feature_extractor import FeatureGenerator
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


