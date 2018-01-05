from PIL import Image
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import cv2
import matplotlib.pyplot as plt

def deconvMask(maskIm,imgIm,tl,kernel,offset):
    rim=np.zeros(imgIm.shape[:2],dtype=imgIm.dtype)
    xlsp = np.linspace(0, tl, tl+1);
    xv, yv = np.meshgrid(xlsp,xlsp)
    xv=xv.astype('uint')
    yv=yv.astype('uint')
    rim[xv*8+offset,yv*8+offset]=maskIm[xv,yv]
    rim=gaussian_filter(rim,kernel)
    mask=rim/float(rim.max())
    img=imgIm/float(imgIm.max() )
    retFrame=img*np.dstack([mask**2]*3)
    return retFrame



mask=np.array(Image.open('1.png').resize([7,7]))[:,:,0]
imgIm=np.array(Image.open('orig.png').convert('RGB'),dtype=np.float32)

img = deconvMask(mask,imgIm,6,4,13)

plt.imsave("1_masked.png", img, format="png")
