
import openslide
import numpy as np


class Slide(OpenSlide):
    """
    WSI object that enables annotation overlay

    wrapper around openslide.OpenSlide class loads WSIs 
    and provides additional functionality to generate 
    masks and mark with user loaded annotations

    Attributes:
        _slide_mask: ndarray mask representation
        dims: dimensions of WSI
        name: string name
        draw_border: boolean to generate border based on annotations
        _border: list of border coordinates [(x1,y1),(x2,y2)] 
    """

    MAG_fACTORS={0:1,1:2,3:4,4:16,5:32}

    def __init__(self, filename, draw_border=False, 
                 annotations=None, annotations_path=None):
        super().__init__(filename)
        

        self.dims = self.dimensions
        self.name = os.path.basename(filename)[:-4]
        self.draw_border=draw_border
        self._border=None

    @property
    def border(self):
        return self._border

    @border.setter
    def border(self,value):
        #Todo: if two values we treat as max_x and max_y
        assert(len(value)==4)

    @draw_border.setter 
    def draw_border(self, value):
        
        if value:
            self._border=self.draw_border()
            self.draw_border=value
        elif not value:
            self._border=[[0,self.dims[0]],[0,self.dims[1]]]
            self.draw_border=value
        else:
            raise TypeError('Boolean type required')
        
    @staticmethod   
    def resize_border(dim, factor=1, threshold=None, operator='=>'):
        """
        resize and redraw annotations border - useful to cut out 
        specific size of WSI and mask

        Args:
            dim: dimensions
            factor: border increments
            threshold: min/max size
            operator: threshold limit 

        Returns:
            new_dims: new border dimensions [(x1,y1),(x2,y2)]

        """

        if threshold is None:
            threshold=dim

        operator_dict={'>':op.gt,'=>':op.ge,'<':op.lt,'=<':op.lt}
        operator=operator_dict[operator] 
        multiples = [factor*i for i in range(100000)]
        multiples = [m for m in multiples if operator(m,threshold)]
        diff = list(map(lambda x: abs(dim-x), multiples))
        new_dim = multiples[diff.index(min(diff))]
       
        return new_dim


    #TODO: function will change with format of annotations
    #data structure accepeted
    def draw_border(self, space=100):
        """
        generate border around annotations on WSI

        Args:
            space: border space
        Returns: 
            self._border: border dimensions [(x1,y1),(x2,y2)]
        """

        coordinates = list(chain(*[self.annotations[a] for a in 
                                   self.annotations]))
        coordinates=list(chain(*coordinates))
        f=lambda x: (min(x)-space, max(x)+space)
        self._border=list(map(f, list(zip(*coordinates))))

        return self._border


    def generate_region(self, mag=0, x=None, y=None, x_size=None, y_size=None, 
                        scale_border=False, factor=1, threshold=None, operator='=>'):
        """
        extracts specific regions of the slide

        Args:
            mag: magnfication level 1-8
            x: 
            y:
            x_size: x dim size
            y_size: y dim size
            scale_border: resize border
            factor: increment for resizing border
            threshold: limit for resizing border
            operator: operator for threshold
        Returns:
            region: ndarray image of extracted region
            mask: ndarray mask of annotations in region

        """

        if x is None:
            self.draw_border()
            x, y = self.border
        
        x_min, x_max=x
        y_min, y_max=y

        x_size=x_max-x_min
        y_size=y_max-y_min

        #Adjust sizes - treating 0 as base
        #256 size in mag 0 is 512 in mag 1
        x_size=int(x_size/Slide.MAG_fACTORS[mag])
        y_size=int(y_size/Slide.MAG_fACTORS[mag])
        
        if scale_border:
            x_size = Slide.resize_border(x_size, factor, threshold, operator)
            y_size = Slide.resize_border(y_size, factor, threshold, operator)
        
        print('x_size:{}'.format(x_size))
        print('y_size:{}'.format(y_size))

        region=self.read_region((x_min,y_min),mag,(x_size, y_size))
        mask=self.slide_mask()[x_min:x_min+x_size,y_min:y_min+y_size]

        return region, mask

    
    def save(self, path, size=(2000,2000), mask=False):
        """
        save thumbnail of slide in image file format
        Args:
            path:
            size:
            mask:
        """

        if mask:
            cv2.imwrite(path,self._slide_mask)
        else: 
            image = self.get_thumbnail(size)
            image = image.convert('RGB')
            image = np.array(image)
            cv2.imwrite(path,image)
