import numpy as np
import openslide
from matplotlib.path import Path

class Patching(Slide):

    MAG_fACTORS={0:1,1:2,3:4,4:16,5:32}

    def __init__(self, slide, annotations, size=(256, 256), mag_level=0,
            border=None, mode=False):
    
        super().__init__()
        self.slide=slide 
        self.mag_level = mag_level
        self.size = size
        self.mode = mode 
        self._number = None
        self._step = step 
        self._patches = []
        self._masks = []
        self._slide_mask = None
        self._class_no = []
        self.__magfactor=mag_factors[self.mag_level]
    

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
    def step(self):
        return self._step


    @step.setter
    def step(self, value):
        step=step*MAG_FACTORS[self.mag_level]
        self._step=step

    
    def patching(self):

        xmin, xmax = border[0][0], border[0][1]
        ymin, ymax = border[1][0], border[1][1]
        for x in range(xmin,xmax, self.step):
            for y in range(ymin,ymax,self.step): 
                yield x, y

#TODO discard patches at borders that do not match size
    def generate_patches(self):
    
        mask=self.slide_mask()

        for p in patching:
            self.patches.append({'x':x,'y':y})
            mask = self.slide_mask[y:y+self.size[0],x:x+self.size[1]]
            classes = dict(zip(*np.unique(mask,return_counts=True)))
            self._class_no.append(len(classes))
            self.masks.append({'x':x, 'y':y, 'classes':classes})

        if self.mode=='focus':
            self.contains()

        return len(self._patches)


    def focus(self):

        index  = [i for i in range(len(self._class_no)) 
                  if self._class_no[i] > 1]

        self._patches = [self.patches[i] for i in index]

        return len(self._patches)

    
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
    
    
    def extract_patch(self, x=None, y=None):
        patch=self.slide.read_region(x,y,self.mag_level,(self.size[0],self.size[1]))
        return patch


    def extract_patches(self):
        for p in self._patches:
            patch=extract_patch(p['x'],p['y'])
            yield patch
    
       
    def extract_mask(self, x=None, y=None):
        mask=slide_mask[y:y+self.size[0],x:x+self.size[1]]
        return mask 
    
    
    def extract_masks(self):
        for p in self._patches:
            mask=self.extract_mask(p['x'],p['y'])
            yield mask



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
