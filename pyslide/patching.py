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

    MAG_FACTORS={0:1,1:2,2:4,3:8,4:16}

    def __init__(self, slide, annotations=None, size=(256, 256),
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
    def mag_factor(self):
        return Patching.MAG_FACTORS[self.mag_level]

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


    def _remove_edge_cases(self,x,y):
        """
        remove edge cases based on dimensions of patch
        :param x: base x coordinate to test 
        :param y: base y coordiante to test
        :return remove: boolean remove patch or not
        """
        x_size=int(self.size[0]*self.mag_factor*.5)
        y_size=int(self.size[1]*self.mag_factor*.5)
        xmin=self.slide._border[0][0]
        xmax=self.slide._border[0][1]
        ymin=self.slide._border[1][0]
        ymax=self.slide._border[1][1]
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
        """
        generate patch coordinates based on mag,step and size
        :param step: integer: step size
        :param mode: sparse or focus
        :param mask_flag: include masks
        :return len(self._patches): Number of patches
        """
        self._patches=[]
        self._masks=[]
        step=step*self.mag_factor
        xmin=self.slide._border[0][0]
        xmax=self.slide._border[0][1]
        ymin=self.slide._border[1][0]
        ymax=self.slide._border[1][1]

        for x, y in self.patching(step,xmin,xmax,ymin,ymax):
            name=self.slide.name+'_'+str(x)+'_'+str(y)
            if self._remove_edge_cases(x,y):
                #continue
                pass
            self.patches.append({'name':name,'x':x,'y':y})
            if mask_flag:
                mask=self.slide.slide_mask[y:y+self.size[0],x:x+self.size[1]]
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
        mask=self.slide.generate_mask()[y-y_size:y+y_size,x-x_size:x+x_size]
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
        print('image_path',image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

            mask_generator=self.extract_masks()
        for patch,x,y in self.extract_patches():
            #if np.mean(patch[:,:,1])>210:
                #continue
            #test=patch[:,:,1]

            #if len(test[test>220])>(0.3*self.size[0]**2):
                #print('here')
                #continue
            patchstatus=self.saveimage(patch,patchpath,self.slide.name,x,y)
            if mask_flag:
                mask,x,y=next(mask_generator)
                maskstatus=self.saveimage(mask,maskpath,self.slide.name,x,y)


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
        z=2048*self.mag_factor
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
