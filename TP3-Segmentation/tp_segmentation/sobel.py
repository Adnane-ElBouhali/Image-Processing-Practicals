#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22 May 2019

@author: M Roux
"""
#%%
import math
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk

from scipy import ndimage
from scipy import signal

from skimage import io

from skimage import filters


##############################################

import mrlab as mr

##############################################"
#%%
############## le close('all') de Matlab
plt.close('all')
################################"
#%%

ima=io.imread('cell.tif')
sigma=0
seuilnorme=0.1

gfima=filters.gaussian(ima,sigma)

plt.figure('Image originale')
plt.imshow(ima, cmap='gray')

plt.figure('Image filtrée (passe-bas)')
plt.imshow(gfima, cmap='gray')
#%%
gradx=mr.sobelGradX(gfima)
grady=mr.sobelGradY(gfima)  

plt.figure('Gradient horizontal')
plt.imshow(gradx, cmap='gray')

plt.figure('Gradient vertical')
plt.imshow(grady, cmap='gray')
#%%
norme=np.sqrt(gradx*gradx+grady*grady)
print(norme)

plt.figure('Norme du gradient')
plt.imshow(norme, cmap='gray')

#%%
direction=np.arctan2(grady,gradx)
    
plt.figure('Direction du Gradient')
plt.imshow(direction, cmap='gray')

#%%
contoursnorme =(norme>seuilnorme) 

plt.figure('Norme seuillée')
plt.imshow(255*contoursnorme)

#%%
contours=np.uint8(mr.maximaDirectionGradient(gradx,grady))

plt.figure('Maxima du gradient dans la direction du gradient')
plt.imshow(255*contours)

#%%
valcontours=(norme>seuilnorme)*contours
      
plt.figure()
plt.imshow(255*valcontours)
plt.show()

#%%
seuils = [-2, -1, 0.01, 0.1, 0.5, 1, 2]

for seuil in seuils:
    seuilnorme = seuil
    contoursnorme =(norme>seuilnorme) 
    
    plt.figure(figsize=(8, 4))

    plt.imshow(255*contoursnorme)
    plt.title(f"Seuil = ({seuil})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
# %%
import tempfile
import IPython

def viewimage(im, normalize=True,titre='',displayfilename=False):
    imin=im.copy().astype(np.float32)
    if normalize:
        imin-=imin.min()
        if imin.max()>0:
            imin/=imin.max()
    else:
        imin=imin.clip(0,255)/255
        
    
    imin=(imin*255).astype(np.uint8)
    filename=tempfile.mktemp(titre+'.png')
    if displayfilename:
        print (filename)
    plt.imsave(filename, imin, cmap='gray')
    IPython.display.display(IPython.display.Image(filename))
# %%
ima=io.imread('pyramide.tif')
viewimage(ima)
# %%
