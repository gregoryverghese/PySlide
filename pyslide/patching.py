#!usr/bin/env python3

import os

import numpy as np
import openslide
import cv2
import seaborn as sns
from matplotlib.path import Path

from PySlide.pyslide import Slide

print(sns.__version__)


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
        self._magfactor=Patching.MAG_FACTORS[self.mag_level]

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


    #TODO discard patches at borders that do not match size
    def generate_patches(self,step, mode='sparse'):
        print(sns.__version__) 
        step=step*self._magfactor
        
        xmin, xmax = self.slide.border[0][0], self.slide.border[0][1]
        ymin, ymax = self.slide.border[1][0], self.slide.border[1][1]

        for x, y in self.patching(step,xmin,xmax,ymin,ymax):

            self.patches.append({'x':x,'y':y})
            mask = self.slide._slide_mask[y:y+self.size[0],x:x+self.size[1]]
           
            #TODO:Do we store that mask class info
            #expensive operation but clearly useful
            if mode=='focus':
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
        print(ratio>=threshold)
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
    


    def extract_patch(self, x=None, y=None):
        patch=self.slide.read_region((x,y),self.mag_level,(self.size[0],self.size[1]))
        patch=np.array(patch.convert('RGB'))
        return patch


    def extract_patches(self):
        for p in self._patches:
            patch=self.extract_patch(p['x'],p['y'])
            yield patch,p['x'],p['y']
    

    #TODO: we need to sort out how we deal with channels
    # when we have binary and multisclass cases
    #right now we assume binary
    def extract_mask(self, x=None, y=None):
        mask=self.slide_mask[y:y+self.size[0],x:x+self.size[1]][:,:,0]
        #mask=self.slide_mask[x:x+self.size[0],y:y+self.size[1]]
        return mask 
    
    
    #generator and next not working as I expected. Learn generators!
    def extract_masks(self):
        for m in self._masks:
            mask=self.extract_mask(m['x'],m['y'])
            yield mask,m['x'],m['y']


    #TODO: how to save individiual patch and mask
    @staticmethod
    def saveimage(image,path,filename,x=None,y=None):
        if (x and not y) or (not x and y):
            raise ValueError('missing value for x or y')
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

        maskpath=os.path.join(path,'masks')
        try:
            os.mkdir(os.path.join(maskpath))
        except OSError as error:
            print(error)
        
        for patch,x,y in self.extract_patches(): 
            patchstatus=self.saveimage(patch,patchpath,self.slide.name,x,y)
            if mask_flag:
                mask,x,y=next(self.extract_masks())
                maskstatus=self.saveimage(mask,maskpath,self.slide.name,x,y)
         

    '''    
    #TODO:check my filter method
    @staticmethod 
    def contains(verts):

        xx,yy=np.meshgrid(np.arange(300),np.arange(300))
        xx,yy=xx.flatten(),yy.flatten()
        verts=np.stack([x,y]).T
        p=Path(verts)
        mask=p.contains_points(verts)
        num=(tolerance*grid.shape[0])
        x = len(grid[grid==True])
        return verts


    #Do we want to use filtering based on orign point
    #or do we want to filter based on all points within patch
    def within(self, boundaries=None):

        if boundaries is None:
            boundaries=self.slide.border

        path = Path(boundaries)
        f = lambda x: p.contains([x['x'],x['y']])
        self_.patches=list(filter(f, self._patches))
    
        return self_patches
    '''


'''
class Stitching():
    def __init__(self, _patches):
        super().__init__()
        self.x = self.patches.dims[0]
        self.y = self.patches.dims[1]
        self.image = np.zeros((self.x, self.y))

    @property
    def slide(self):
        return self._slide
 

    def stitch():
        
        temp=np.zeros((int(h), int(w, 3))
        for p in self.extract_patches()
            self.image[ynew:ynew+ysize,xnew:xnew+xsize,0]=p[:,:,0]
            self.image[ynew:ynew+ysize,xnew:xnew+xsize,1]=p[:,:,1]
            self.image[ynew:ynew+ysize,xnew:xnew+xsize,1]=p[:,:,1]
        
        return image
'''
