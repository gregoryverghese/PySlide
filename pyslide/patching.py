#!usr/bin/env python3

"""
patching.py: contains Patching class for splitting WSIs
into a set of smaller tiles based on annotations
"""

import os
import glob 

import numpy as np
import openslide
import cv2
import seaborn as sns
from matplotlib.path import Path

from pyslide.slide import Slide

__author__='Gregory Verghese'
__email__='gregory.verghese@gmail.com'


class Patching():

    MAG_FACTORS={0:1,1:2,3:4,4:16,5:32}

    def __init__(self, slide, annotations, size=(256, 256), 
                 mag_level=0,border=None, mode=False):
    
        super().__init__()
        self.slide=slide 
        self.mag_level=mag_level
        self.size=size
        self._number=None
        self._patches=[]
        self._masks=[]

    @property
    def masks(self):
        return self._masks
    
    @property
    def patches(self):
        return self._patches

    @property
    def annotations(self):
        return _self.annotations

    @property
    def mag_factor(self):
        return Patching.MAG_FACTORS[self.mag_level]

    @property
    def slide_mask(self):
        return self.slide._slide_mask
    
    @property
    def config(self):
        config={'name':self.slide.name,
                'mag':self.mag_level,
                'size':self.size,
                'border':self.slide.border,
                'mode':None,
                'number':self._number}
        return config
     

    def __repr__(self):
        return str(self.config)


    @staticmethod
    def patching(step,xmin,xmax,ymin,ymax):
        for x in range(xmin,xmax, step):
            for y in range(ymin,ymax,step):
                yield x, y


    def _remove_edge_cases(self,x,y,xmin,xmax,ymin,ymax):
        x_size=int(self.size[0]*self.mag_factor*.5)
        y_size=int(self.size[1]*self.mag_factor*.5)
        xmin=self.slide.border[0][0]
        xmax=self.slide.border[0][1]
        ymin=self.slide.border[1][0]
        ymax=self.slide.border[1][1]
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


    def generate_patches(self,step, mode='sparse',mask_flag=False):
        self._patches=[]
        self._masks=[]
        step=step*self.mag_factor
        xmin=self.slide.border[0][0]
        xmax=self.slide.border[0][1]
        ymin=self.slide.border[1][0]
        ymax=self.slide.border[1][1]

        for x, y in self.patching(step,xmin,xmax,ymin,ymax):
            name=self.slide.name+'_'+str(x)+'_'+str(y)
            if self._remove_edge_cases(x,y):
                continue
            self.patches.append({'name':name,'x':x,'y':y})
            if mask_flag:
                mask=self.slide._slide_mask[y:y+self.size[0],x:x+self.size[1]]
                if mode == 'focus':
                    classes = len(np.unique(mask))
                    self._masks.append({'x':x, 'y':y, 'classes':classes})
                    self.focus()
                else:
                    self._masks.append({'x':x, 'y':y})

        self._number=len(self._patches)
        return self._number
    

    def focus(self, task='classes'):
        
        if task=='classes':
            index=[i for i in range(len(self._patches)) if
                  self._masks[i][task] >1]
        elif task=='labels':
            index=[i for i in range(len(self._patches)) if
                   self._masks[i][task]!=9]

        self._patches = [self.patches[i] for i in index]
        self._masks = [self.masks[i] for i in index]

        return len(self._patches)

    
    @staticmethod
    def __filter(y_cnt,cnts,threshold):
        ratio=y_cnt/float(sum(cnts))
        return ratio>=threshold

    
    #TODO:how do we set a threshold in multisclass
    def generate_labels(self,threshold=1):
        labels=[]
        for i, (m,x,y) in enumerate(self.extract_masks()):
            cls,cnts=np.unique(m, return_counts=True)
            y=cls[cnts==cnts.max()]
            y_cnt=cnts.max()
            if self.__filter(y_cnt,cnts,threshold): 
                self.masks[i]['labels']=y[0]
                labels.append(y)
            else:
                self.masks[i]['labels']=9 
                #TODO:do we want a labels attribute
                labels.append(y)

        return np.unique(np.array(labels),return_counts=True)
            

    def plotlabeldist(self):
        labels=[self.masks[i]['labels'] for i in range(len(self.masks))]
        return sns.distplot(labels)
    

    #TODO: maybe we don't need .5 - should check 
    def extract_patch(self, x=None, y=None):
        x_size=int(self.size[0]*self.mag_factor*.5)
        y_size=int(self.size[1]*self.mag_factor*.5)

        patch=self.slide.read_region((x-x_size,y-y_size), self.mag_level,
                                     (self.size[0],self.size[1]))
        patch=np.array(patch.convert('RGB'))
        return patch


    def extract_patches(self):
        for p in self._patches:
            patch=self.extract_patch(p['x'],p['y'])
            yield patch,p['x'],p['y']
    

    def extract_mask(self, x=None, y=None):

        x_size=int(self.size[0]*self.mag_factor*.5)
        y_size=int(self.size[1]*self.mag_factor*.5)
        mask=self.slide_mask[y-y_size:y+y_size,x-x_size:x+x_size][:,:,0]
        mask=cv2.resize(mask,(self.size[0],self.size[1]))

        return mask 
    
    
    def extract_masks(self):
        for m in self._masks:
            mask=self.extract_mask(m['x'],m['y'])
            yield mask,m['x'],m['y']


        #TODO: how to save individiual patch and mask
    @staticmethod
    def saveimage(image,path,filename,x=None,y=None):

        if y is None and x is not None:
            raise ValueError('missing y')
        elif x is None and y is not None:
            raise ValueError('missing x')
        elif (x and y) is None:
            image_path=os.path.join(path,filename)
        elif (x and y) is not None:
             filename=filename+'_'+str(x)+'_'+str(y)+'.png'
             image_path=os.path.join(path,filename)
        status=cv2.imwrite(image_path,image)
        return status
   

    #TODO fix masks. Currently saving only first mask over and over
    def save(self, path, mask_flag=False):
    
        patchpath=os.path.join(path,'images')
        try:
            os.mkdir(patchpath)
        except OSError as error:
            print(error)
    
        if mask_flag:
            maskpath=os.path.join(path,'masks')
            try:
                os.mkdir(os.path.join(maskpath))
            except OSError as error:
                print(error)

            masks_generator=self.extract_masks()
        for patch,x,y in self.extract_patches(): 
            patchstatus=self.saveimage(patch,patchpath,self.slide.name,x,y)
            if mask_flag:    
                mask,x,y=next(mask_generator)
                maskstatus=self.saveimage(mask,maskpath,self.slide.name,x,y)


class Stitching():
    def __init__(self,patch_path,slide=None,name=None,step=None,dims=None):
        self.patch_path=patch_path
        self.coords=self._get_coords()
        self.name=name
        self.step=self._get_step() if step is None else step
    
        if dims is not None:
            self.dims=dims
        elif slide is not None:
            self.dims=slide.dims
        else:
            self.dims=self._get_dims()


    def _get_coords(self):
        patch_files=glob.glob(os.path.join(self.patch_path,'*'))
        print('found {} patches'.format(len(patch_files)))
        coords=[(int(f.split('_')[-2:][0]),int(f.split('_')[-2:][1][:-4])) 
                for f in patch_files]
        
        self._coords=coords
        return self._coords


    def _get_dims(self):
        coords=self._get_coords()
        x_dim=max([c[0] for c in coords])
        y_dim=max([c[1] for c in coords])

        return (x_dim,y_dim)


    def _get_step(self):
        coords=self._get_coords()
        xs=[c[0] for c in coords]
        step=min([abs(x1-x2) for x1, x2 in zip(xs, xs[1:]) if abs(x1-x2)!=0])
    
        return step


    def stich(self):

        xmin=self.dims[0][0]
        xmax=self.dims[0][1]
        ymin=self.dims[1][0]
        ymax=self.dims[1][1]
        #temp=np.array

        for x in range(xmin,xmax,step):
            for y in range(ymin,ymax,step):
                filename=self.name+'_'+str(x)+'_'+str(y)+'.png'
                p=cv2.imwrite(filename)
                temp[y:y+step,x:x+step,0]=p[:,:,0]
                temp[y:y+step,x:x+step,1]=p[:,:,1]
                temp[y:y+step,x:x+step,2]=p[:,:,2]
        return temp










