"""
patching.py: contains 1. Patching class 2. Stitching class.

Patching class - takes WSI openslide slide object and  defines patching
functions to generate a set of tiles. Control magnfication, step size, 
and patch size. Generate patch binary/multiclass masks or patch-level lavels  
based on annotations. 

Stitching class - takes location of WSI patches with a filename format filename_x_y_.png 
and will stitch tiles to form entire slide or mask representation.
"""

import os
import glob
import json
import random

import numpy as np
import cv2
import seaborn as sns
from matplotlib.path import Path
from openslide import OpenSlide
import pandas as pd
import seaborn as sns
from itertools import chain
import operator as op

from pyslide.util.utilities import mask2rgb
from pyslide.exceptions import StitchingMissingPatches
from pyslide.analysis.filters import entropy
from pyslide.io.lmdb_io import LMDBWrite

__author__='Gregory Verghese'
__email__='gregory.verghese@gmail.com'


class Patch():
    def __init__(self, 
                 slide, 
                 size, 
                 mag_level=0,
                 border=None,  
                 step=None):

        super().__init__()
        self.slide=slide
        self.mag_level=mag_level
        self.size=size
        self.border=slide._border if border is None else border
        self._x_min = int(self.border[0][0])
        self._x_max = int(self.border[0][1])
        self._y_min = int(self.border[1][0])
        self._y_max = int(self.border[1][1])
        self.step=size[0] if step is None else step
        #self.mode='sparse' if mode is None else mode
        self._patches=[]
        self._labels=[]
        self._downsample=int(slide.level_downsamples[mag_level])
        num=self.generate_patches(self.step)
        print('num patches: {}'.format(num))
        

    @property
    def number(self):
        return len(self._patches)


    @property
    def patches(self):
        return self._patches


    @patches.setter
    def patches(self,value):
        self._patches=patches


    @property
    def label(self):
        return self._labels


    @property
    def config(self):
        config={'name':self.slide.name,
                'mag':self.mag_level,
                'size':self.size,
                'step':self.step,
                'border':self.border,
                'mode':None,
                'number':self._number}
        return config


    def __repr__(self):
        """
        object representation
        :return str(self.config)
        """
        return str(self.config)


    def _patching(self,step):
        """
        step across coordinate range
        """
        for x in range(self._x_min,self._x_max, step):
            for y in range(self._y_min,self._y_max,step):
                yield x, y


    def _remove_edge_case(self,x,y):
        """
        remove edge cases based on dimensions of patch
        :param x: base x coordinate to test 
        :param y: base y coordiante to test
        :return remove: boolean remove patch or not
        """
        x_size=int(self.size[0]*self._downsample)
        y_size=int(self.size[1]*self._downsample)
        remove=False
        if x+x_size>self._x_max:
            remove=True
        if y+y_size>self._y_max:
            remove=True
        return remove


    def generate_patches(self, 
                         step, 
                         edge_cases=False):
        """
        generate patch coordinates based on mag,step and size
        :param step: integer: step size
        :param mode: sparse or focus
        :param mask_flag: include masks
        :return len(self._patches): Number of patches
        """
        self.step=step
        self._patches=[]
        step=step*self._downsample

        if (self._x_max,self._y_max)==self.slide.dims:
            edge_cases==True
        for x, y in self._patching(step):
            name=self.slide.name+'_'+str(x)+'_'+str(y)
            if edge_cases:
                if self._remove_edge_case(x,y):
                    continue
            self._patches.append({'name':name,'x':x,'y':y})

        self._number=len(self._patches)
        return self._number


    def focus(self, num=2):
        """
        remove patches with no classes
        :param num: number of classes required
        :return len(self._patches): number of patches
        """
        for p in self._patches:
            if self.slide.annotations is None:
                x,y =(p['x'],p['y'])
                mask=self.slide.slide_mask[y:y+self.size[0],x:x+self.size[1]]
                classes = len(np.unique(mask))
                if classes<num:
                    self._patches.remove(p)
        return len(self._patches)


    @staticmethod
    def _filter(y_cnt,cnts,threshold):
        """
        check proportion of class count
        :param y_cnt: pixel count of class
        :param cnts: total number of pixels
        :param threshold: threshold proportion
        :return boolean if class count > threshold
        """
        ratio=y_cnt/float(sum(cnts))
        return ratio>=threshold
    

    #TODO: how to treat labels that don't pass
    #threshold test
    def generate_labels(self,threshold=0.5):
        """
        generate patch labels based on pixel-level annotations
        :param threshold: threshold proportion
        :return classes and count
        """
        #empty annotations
        self._labels=[]
        if self.slide.annotations is None: 
            self._labels=[np.nan]*len(self._patches)

        for i, (mask,_) in enumerate(self.extract_masks()):
            cls,cnts=np.unique(mask, return_counts=True)
            cls,cnts=(list(cls),list(cnts))
            cnts.pop(cls.index(0))
            cls.remove(0)
            y=cls[cnts.index(max(cnts))]
            y_cnt=max(cnts)

            if self._filter(y_cnt,cnts,threshold):
                self._patches[i]['label']=y
                self._labels.append(y)
            else:
                self._patches[i]['label']=np.nan
                self._labels.append(np.nan)

        return np.unique(np.array(self._labels),return_counts=True)


    def filter_labels(self):
        for i,l in enumerate(self.labels):
            self._patches.pop(i)
            self._labels.pop(i)
        return len(self._patches)

    
    def plot_class_dist(self):
        """
        plot label distribution
        :return sns.distplot for classes
        """
        #Raise error for no labels calculated yet
        if len(self._labels)==0:
            self.generate_labels()
        cls,cnts=np.unique(np.array(self._labels),return_counts=True)
        return sns.barplot(x=cls,y=cnts)


    def filter_patches(self,
                       filter_type,
                       threshold,
                       channel=None):
        """
        filter patches based on pixel intensity
        :param threshold: intesnity threshold value
        :param channel: channel index
        :return removed: number of removed
        """
        num_b4=self._number
        patches=self._patches.copy()

        if filter_type=='entropy':
            for patch, p in self.extract_patches():
                avg_entropy=entropy(patch)
            if avg_entropy<threshold:
                self._patches.remove(p)

        elif filter_type=='intensity':
            if channel is not None:
                for patch,p in self.extract_patches():
                    if np.mean(patch[:,:,channel])>threshold:
                        self._patches.remove(p)
            elif channel is None:
                for patch,p in self.extract_patches():
                    if np.mean(patch)>threshold:
                        patches.remove(p)
                        continue

        self._patches=patches.copy()
        removed=num_b4-len(self._patches)
        print('Num removed: {}'.format(removed))
        print('Remaining:{}'.format(len(self._patches)))

        return removed



    def extract_patch(self, x=None, y=None):
        """
        extract individual patch from WSI
        :param x: int x coordinate
        :param y: int y coodinate
        :return patch: ndarray patch
        """
        #if we want x,y coordinate of point to be central
        #points in read_region (x-x_size,y-y_size)
        #x_size=int(self.size[0]*self.mag_factor*.5)
        #y_size=int(self.size[1]*self.mag_factor*.5)
        patch=self.slide.read_region((x,y), self.mag_level,
                                     (self.size[0],self.size[1]))
        patch=np.array(patch.convert('RGB'))
        return patch


    def extract_patches(self):
        """
        generator to extract all patches
        :yield patch: ndarray patch
        :yield p: patch dict metadata
        """
        for p in self._patches:
            patch=self.extract_patch(p['x'],p['y'])
            yield patch, p


    def extract_mask(self, x=None, y=None):
        """
        extract binary mask corresponding to patch
        from slide_mask
        :param x: int x coordinate
        :param y: int y coordinate
        """
        #if we want x,y coordinate of point to be central
        #x_size=int(self.size[0]*self.mag_factor*.5)
        #y_size=int(self.size[1]*self.mag_factor*.5)
        #[y-y_size:y+y_size,x-x_size:x+x_size]
        x_size=int(self.size[0]*self._downsample)
        y_size=int(self.size[1]*self._downsample)
        mask=self.slide.generate_mask()[y:y+y_size,x:x+x_size]
        mask=cv2.resize(mask,(self.size[0],self.size[1]))
        return mask


    def extract_masks(self):
        """
        extract all masks
        :yield mask: ndarray mask
        :yield m: mask dict metadata
        """
        for m in self._patches:
            mask=self.extract_mask(m['x'],m['y'])
            yield mask,m


    @staticmethod
    def _save_disk(image,path,filename,x=None,y=None):
        """
        save patch
        :param image: ndarray patch
        :param path: path to save
        :param filename: str filename
        :param x: int x coordindate for filename
        :param y: int y coordinate for filename
        """
        if y is None and x is not None:
            raise ValueError('missing y')
        elif x is None and y is not None:
            raise ValueError('missing x')
        elif (x and y) is None:
            image_path=os.path.join(path,filename)
        elif (x and y) is not None:
             filename=filename+'_'+str(x)+'_'+str(y)+'.png'
             image_path=os.path.join(path,filename)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        status=cv2.imwrite(image_path,image)
        return status
   

    def save_masks(self,path,dir_name):

        mask_generator=self.extract_masks()
        mask_path=os.path.join(path,dir_name)
        os.makedirs(mask_path,exist_ok=True)
        filename=self.slide.name
        for mask,m in self.extract_masks():
            self._save_disk(mask,mask_path,filename,m['x'],m['y'])


    def save(self, 
             path, 
             mask_flag=False, 
             label_dir=False, 
             label_csv=False):
        """
        object save method. saves down all patches
        :param path: save path
        :param masK_flag: boolean to save masks
        :param label_dir: label directory
        :param label_csv: boolean to save labels in csv
        """
        patch_path=os.path.join(path,'images')
        os.makedirs(patch_path,exist_ok=True)
        filename=self.slide.name

        for patch,p in self.extract_patches():
            if label_dir:
                 patch_path=os.path.join(path_path,patch['labels'])
            self.save_image(patch,patch_path,filename,p['x'],p['y'])
        if mask_flag:
            mask_generator=self.extract_masks()
            mask_path=os.path.join(path,'masks')
            os.makedirs(mask_path,exist_ok=True)
            if label_dir:
                patch_path=os.path.join(path_path,patch['labels'])
                for mask,m in self.extract_masks():
                    self.save_image(mask,mask_path,filename,m['x'],m['y'])

        if label_csv:
            df=pd.DataFrame(self._patches,columns=['names','x','y','labels'])
            df.to_csv(os.path.join(path,'labels.csv'))


    def to_lmdb(self, db_path, write_frequency=100):
        size_estimate=len(self._patches)*self.size[0]*self.size[1]*3
        print(size_estimate/1e6)
        db_write=LMDBWrite(db_path,size_estimate,write_frequency)
        db_write.write(self)
            

    def to_tfrecords(self, db_path):
        pass


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
