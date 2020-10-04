#from patch import Patching
from utilities import getRegions 
import openslide
from patch import Patching

classKey = {'SINUS':1}

annotations = getRegions('U_100188_10_X_HIGH_10_L1.xml')

keys = annotations.keys()
for k in list(keys):
    if k not in classKey:
        del annotations[k]

annotations = {classKey[k]: [v2['coords'] for k2, v2 in v.items()] for k,v in annotations.items()}

slide = openslide.OpenSlide('U_100188_10_X_HIGH_10_L1.ndpi')
p=Patching(slide, annotations, boundaries='draw')

patches=p.extract_patches()


print(len(patches), sys.getsizeof(patches))
print(patches[0])
cv2.imwrite('test.png', np.array(patches[10000]))

