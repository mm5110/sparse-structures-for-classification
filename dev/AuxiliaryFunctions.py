
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np

def showFilters(W,ncol,nrows):
    # Display filters
    p = int(W.shape[3])+2
    Nimages = W.shape[0]
    Mosaic = np.zeros((p*ncol,p*nrows))
    indx = 0
    for i in range(ncol):
        for j in range(nrows):
            im = W[indx,0,:,:]
            im = (im-np.min(im))
            im = im/np.max(im)
            Mosaic[ i*p : (i+1)*p , j*p : (j+1)*p ] = np.pad(im.reshape(W.shape[3],W.shape[3]),(1,1),mode='constant').reshape(p,p)
            indx += 1
            
    return Mosaic