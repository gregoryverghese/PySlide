import numpy as np


class Preprocessing():
    def __init__(self):
        self.masks = mask
        self.patches = patch
        self._weights = None

    @property
    def weights(self):
        return self_weights



    def image_std(image):

        channel_values = np.sum(image, axis=(0,1), dtype='float64')
        shape = image.shape

        channel_values_sq= no.sum(np.square(image-mean), axis=(0,1),
                                  dtype='float64')

        std = np.sqrt(channel_values_sq/pixel_num, dtype='float64')

        return std


    @staticmethod
    def image_mean(image):

        channel_values = np.sum(image, axis=(0,1), dtype='float64')
         #TODO calculate inverse frequency of pixels and compare
    def calculate_weights(self, no_classes):
    
        total = {c:0 for c in range(num_classes)}

        for m in self.masks:
            labels = m.reshape(-1)
            classes = np.unique(labels, return_counts=True)

            pixel_dict = dict(list(zip(*classes))) 
    
            for k, v in pixel_dict.items():
                total[k] = total[k] + v 
    
        if num_classes==2:
            self._weights = total[0]/total[1]
        else:
            self._weights = [1/v for v in list(total.values())]

        return self_weights   shape = image.shape

        pixel_num = image_shape[0]*image_shape[1]
        mean=channel_values/pixel_num

        return mean, channel_values, pixel_num


    def calculate_mean(self):
        if channel:
            pass

        for p in self._patches:
            p=p.astype('float64')
            mean=image_mean

        return None


    def calculate_std(self):

        if channel:
            pass

        for p in self._patches:
            p=p.astype('float64')
            std = image_std

        return None


    #TODO calculate inverse frequency of pixels and compare
    def calculate_weights(self, no_classes):
    
        total = {c:0 for c in range(num_classes)}

        for m in self.masks:
            labels = m.reshape(-1)
            classes = np.unique(labels, return_counts=True)

            pixel_dict = dict(list(zip(*classes))) 
    
            for k, v in pixel_dict.items():
                total[k] = total[k] + v 
    
        if num_classes==2:
            self._weights = total[0]/total[1]
        else:
            self._weights = [1/v for v in list(total.values())]

        return self_weights
