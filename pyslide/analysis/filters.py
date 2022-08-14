from skimage.morphology import disk
from skimage.filters.rankk import entropy


def entropy(patch):
    patch=np.array(patch.convert('RGB'))
    gray=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY)
    entr=entropy(gray,disk(10))
    avg_entr=np.mean(entr)
    return avg_entr
