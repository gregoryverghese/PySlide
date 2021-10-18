import os
import glob
import json

import numpy as np
import cv2
import seaborn as sns
from matplotlib.path import Path
from openslide import OpenSlide
import pandas as pd
import seaborn as sns
from itertools import chain
import operator as op
from pyslide.utilities import mask2rgb


__author__='Gregory Verghese'
__email__='gregory.verghese@gmail.com'


class Patching():
    def __init__(self, slide, size, mag_level=0,
                 border=None, mode=None, step=None):

        super().__init__()
        self.slide=slide
        self.mag_level=mag_level
        self.size=size
        self.step=size[0] if step is None else step
        self.mode='sparse' if mode is None else mode
        self._patches=[]
        self._labels=[]
        self._downsample=int(slide.level_downsamples[mag_level])
        num=self.generate_patches(self.step,self.mode)
        print('num patches: {}'.format(num))
    
    @property
    def number(self):
        return len(self._patches)


    @property
    def patches(self):
        return self._patches


    @property
    def label(self):
        return self._labels


    @property
    def config(self):
        config={'name':self.slide.name,
                'mag':self.mag_level,
                'size':self.size,
                'step':self.step,
                'border':self.slide._border,
                'mode':None,
                'number':self._number}
        return config


    def __repr__(self):
        """
        object representation
        :return str(self.config)
        """
        return str(self.config)


    @staticmethod
    def patching(step,xmin,xmax,ymin,ymax):
        """
        step across coordinate range
        """
        for x in range(xmin,xmax, step):
            for y in range(ymin,ymax,step):
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
        xmin=int(self.slide._border[0][0])
        xmax=int(self.slide._border[0][1])
        ymin=int(self.slide._border[1][0])
        ymax=int(self.slide._border[1][1])
        remove=False

        if x+x_size>xmax:
            remove=True
        if x-x_size<xmin:
            remove=True
        if y+y_size>ymax:
            remove=True
        if y-y_size<ymin:
            remove=True
        return remove


    def generate_patches(self, step, mode="Sparse"):
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
        xmin=int(self.slide._border[0][0])
        xmax=int(self.slide._border[0][1])
        ymin=int(self.slide._border[1][0])
        ymax=int(self.slide._border[1][1])
        for x, y in self.patching(step,xmin,xmax,ymin,ymax):
            name=self.slide.name+'_'+str(x)+'_'+str(y)
            if self._remove_edge_case(x,y):
                continue
            self._patches.append({'name':name,'x':x,'y':y})
        if mode=="focus":
            self.focus()
        self._number=len(self._patches)
        return self._number


    def focus(self, num=2):    
        for p in self._patches:
            x=p['x']
            y=p['y'] 
            mask=self.slide.slide_mask[y:y+self.size[0],x:x+self.size[1]]
            classes = len(np.unique(mask))
            if classes<num:
                self._patches.remove(p)
        return len(self._patches)


    @staticmethod
    def __filter(y_cnt,cnts,threshold):
        ratio=y_cnt/float(sum(cnts))
        return ratio>=threshold

     
    def generate_labels(self,threshold=0.5):
        for i, (mask,_) in enumerate(self.extract_masks()):
            cls,cnts=np.unique(mask, return_counts=True)
            cls,cnts=(list(cls),list(cnts))
            if len(cls)>1:
                cnts.pop(cls.index(0))
                cls.remove(0)
            y=cls[cls.index(max(cls))]
            y_cnt=max(cnts)
            if len(cls)>1:
                if self.__filter(y_cnt,cnts,threshold):
                    self._patches[i]['labels']=y
                    self._labels.append(y)
                else:
                    self._patches[i]['labels']=None
                    self._labels.append(None)
            else:
                self._patches[i]['labels']=y
                self._labels.append(y)
        return np.unique(np.array(self._labels),return_counts=True)
    

    def plotlabeldist(self):
        labels=[self.masks[i]['labels'] for i in range(len(self.masks))]
        return sns.distplot(labels)


    def filter_patches(self,threshold,channel=None):
        num_b4=self._number
        patches=self._patches.copy()
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
        #if we want x,y coordinate of point to be central
        #points in read_region (x-x_size,y-y_size)
        #x_size=int(self.size[0]*self.mag_factor*.5)
        #y_size=int(self.size[1]*self.mag_factor*.5)
        patch=self.slide.read_region((x,y), self.mag_level,
                                     (self.size[0],self.size[1]))
        patch=np.array(patch.convert('RGB'))
        return patch


    def extract_patches(self):
        for p in self._patches:
            patch=self.extract_patch(p['x'],p['y'])
            yield patch, p


    def extract_mask(self, x=None, y=None):
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
        for m in self._patches:
            mask=self.extract_mask(m['x'],m['y'])
            yield mask,m


    @staticmethod
    def save_image(image,path,filename,x=None,y=None):

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
    

    def save(self, path, mask_flag=False, label_dir=False, label_csv=False):

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


class Stitching():

    MAG_FACTORS={0:1,1:2,2:4,3:8,4:16}

    def __init__(self,patch_path,slide=None,patching=None,name=None,
             step=None,border=None,mag_level=0):

        self.patch_path=patch_path
        patch_files=glob.glob(os.path.join(self.patch_path,'*'))
        print('found {} patches'.format(len(patch_files)))
        self.fext=patch_files[0].split('.')[-1]
        self.slide=slide
        self.coords=self._get_coords()

        if patching is not None:
            self.name=self.patching.slide.name
        elif slide is not None:
            self.name=self.slide.name
        elif name is not None:
            self.name=name
        else:
            self.name='pyslide_wsi'

        if border is not None:
            self.border=border
        elif patching is not None:
            self.border=patching.slide.border
        elif slide is not None:
            self.border=slide.border
        else:
            self.border=self._get_border()
        print('border',self.border)

        if patching is not None:
            self.mag_level=patching.mag_level
        else:
            self.mag_level=mag_level

        self.step=self._get_step() if step is None else step


    @property
    def mag_factor(self):
         return Stitching.MAG_FACTORS[self.mag_level]


    def _get_coords(self):
        patch_files=glob.glob(os.path.join(self.patch_path,'*'))
        coords=[(int(f.split('_')[-2:][0]),int(f.split('_')[-2:][1][:-4]))
                for f in patch_files]

        self._coords=coords
        return self._coords


    def _get_border(self):
        coords=self._get_coords()
        xmax=max([c[0] for c in coords])
        xmin=min([c[0] for c in coords])
        ymax=max([c[1] for c in coords])
        ymin=min([c[1] for c in coords])

        return [[xmin,xmax],[ymin,ymax]]


    def _get_step(self):
        coords=self._get_coords()
        xs=[c[0] for c in coords]
        step=min([abs(x1-x2) for x1, x2 in zip(xs, xs[1:]) if abs(x1-x2)!=0])
        return int(step/self.mag_factor)


    def stitch(self,size=None):
        step=self.step*self.mag_factor
        xmin,xmax=self.border[0][0],self.border[0][1]
        ymin,ymax=self.border[1][0],self.border[1][1]
        z=self.step*self.mag_factor
        xnew=(xmax+z-xmin)/self.mag_factor
        ynew=(ymax+z-ymin)/self.mag_factor
        canvas=np.zeros((int(ynew),int(xnew),3))
        step=self.step*self.mag_factor
        for x in range(xmin,xmax+step,step):
            for y in range(ymin,ymax+step,step):
                filename=self.name+'_'+str(x)+'_'+str(y)+'.'+self.fext
                p=cv2.imread(os.path.join(self.patch_path,filename))
                xsize,ysize,_=p.shape
                xnew=int((x-xmin)/self.mag_factor)
                ynew=int((y-ymin)/self.mag_factor)
                canvas[ynew:ynew+ysize,xnew:xnew+xsize,:]=p
        return canvas.astype(np.uint8)
