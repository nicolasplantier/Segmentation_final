# Importations 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.segmentation import watershed, inverse_gaussian_gradient
from skimage import data
from skimage import filters
from skimage.filters import rank
from skimage.util import img_as_ubyte
import time
import codecs


#global variables
exposant = 1.
r_denoised = 2.
r_markers = 10.  # for 4 with 1200 resolution 
lim_markers = 30.
r_gradient = 5. # for 4 with 1200 resolution

#adjustement functions
def normalize(array):
    return array/array.max()

def to_255(array):
    return np.uint8(normalize(array)*255)

def for_watershed(array, exposant):
    array = normalize(array)
    array = np.power(array,exposant)
    return normalize(array)

if __name__ == "__main__":

    # Start the clock 
    start = time.time()

    # importation champ scalaire projeté 2D
    filename = 'table_image2.txt'
    array = np.loadtxt(filename)

    ######### v1 without inverse_gaussian_gradient ###########
    """
    array = for_watershed(array, exposant)

    # denoise image
    denoised = rank.median(to_255(array), disk(r_denoised))
    # find continuous region (low gradient -
    # where less than 10 for this image) --> markers
    # disk(5) is used here to get a more smooth image
    markers = rank.gradient(denoised, disk(r_markers)) < lim_markers
    markers = ndi.label(markers)[0]

    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(denoised, disk(r_gradient))

    # process the watershed
    labels = watershed(gradient, markers)
    """
    ######### v1 end #########################################

    ########### v2 with invers_gaussian_gradient##############

    # new gradient
    gradient = 1 - inverse_gaussian_gradient(array, alpha = 1000, sigma = 10)
    gradient = rank.mean(to_255(gradient), disk(5))

    #find best threshold for markers
    def components_sizes(t):
        markers = gradient < t
        markers = ndi.label(markers)[0]
        return [(markers == i).sum() for i in range(markers.max())]
    candidates = range(100,140,2)
    criterions = [sorted(components_sizes(t))[-5]/1000 for t in candidates]
    threshold = candidates[np.argmax(criterions)]
    print(threshold)

    #watersheding
    markers = gradient < threshold
    markers = ndi.label(markers)[0]
    labels = watershed(gradient, markers)

    ########## v2 end #######################################

    #select the biggest ones that might be a tetrapod
    labels_sizes = np.array([(labels == i).sum() for i in range(labels.max())])
    tetra_indices = list(np.where(labels_sizes>50000)[0])
    # tetra_indices = list(np.where(labels_sizes>30000)[0])
    # list of arrays representing masks
    tetrapods = []
    for i in tetra_indices:
        shadow = (labels == i)
        #get rid of the background morceaux
        if (shadow[0,0] != 1) & (shadow[0,-1] != 1) & (shadow[-1,-1] != 1) & (shadow[-1,0] != 1):
            tetrapods.append(shadow)
    for i in range(len(tetrapods)):
        np.savetxt(filename[:-4] + f'tetramask{i}', tetrapods[i])


    # What time is it ? 
    print(np.round(time.time() - start, 2))