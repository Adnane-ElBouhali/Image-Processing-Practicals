#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22 May 2019

@author: M Roux
"""
#%%
############## le clear('all') de Matlab
for name in dir():
    if not name.startswith('_'):
        del globals()[name]
################################"

import math
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk

from scipy import ndimage
from scipy import signal

from skimage import io

##############################################

import mrlab as mr

##############################################"

############## le close('all') de Matlab
plt.close('all')
################################"

#%%
ima=io.imread('cell.tif')
alpha=0.5

gradx=mr.dericheGradX(mr.dericheSmoothY(ima,alpha),alpha)
grady=mr.dericheGradY(mr.dericheSmoothX(ima,alpha),alpha)  

gradx2=mr.dericheGradX(mr.dericheSmoothY(gradx,alpha),alpha)
grady2=mr.dericheGradY(mr.dericheSmoothX(grady,alpha),alpha)  

plt.figure('Image originale')
plt.imshow(ima, cmap='gray')

#%%
lpima=gradx2+grady2

plt.figure('Laplacien')
plt.imshow(lpima, cmap='gray')

#%%
posneg=(lpima>=0)

plt.figure('Laplacien binarisé -/+')
plt.imshow(255*posneg, cmap='gray')
#%%
nl,nc=ima.shape
contours=np.uint8(np.zeros((nl,nc)))

for i in range(1,nl):
    for j in range(1,nc):
        if (((i>0) and (posneg[i-1,j] != posneg[i,j])) or
            ((j>0) and (posneg[i,j-1] != posneg[i,j]))):
            contours[i,j]=255
            
   
plt.figure('Contours')
plt.imshow(contours, cmap='gray')
              
#io.imsave('contours.tif',np.uint8(255*valcontours))
#%%
from skimage import filters
#sigma = 0
ima=io.imread('pyra-gauss.tif')
#ima=filters.gaussian(ima,sigma)
Alphas = [0.5]

for Alpha in Alphas:
    alpha= Alpha

    gradx=mr.dericheGradX(mr.dericheSmoothY(ima,alpha),alpha)
    grady=mr.dericheGradY(mr.dericheSmoothX(ima,alpha),alpha)  

    gradx2=mr.dericheGradX(mr.dericheSmoothY(gradx,alpha),alpha)
    grady2=mr.dericheGradY(mr.dericheSmoothX(grady,alpha),alpha) 
    
    lpima=gradx2+grady2
    posneg=(lpima>=0)
    nl,nc=ima.shape
    contours=np.uint8(np.zeros((nl,nc)))

    for i in range(1,nl):
        for j in range(1,nc):
            if (((i>0) and (posneg[i-1,j] != posneg[i,j])) or
                ((j>0) and (posneg[i,j-1] != posneg[i,j]))):
                contours[i,j]=255

    

    plt.figure(figsize=(8, 4))
    plt.figure('Contours')
    plt.imshow(contours, cmap='gray')
    plt.title(f"Alpha = ({alpha})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
  

# %%
ima=io.imread('pyramide.tif')
plt.figure('Image originale')
plt.imshow(ima, cmap='gray')
# %%
def filtergauss(im):
    """applique un filtre passe-bas gaussien. coupe approximativement a f0/4"""
    (ty,tx)=im.shape
    imt=np.float32(im.copy())
    pi=np.pi
    XX=np.concatenate((np.arange(0,tx/2+1),np.arange(-tx/2+1,0)))
    XX=np.ones((ty,1))@(XX.reshape((1,tx)))
    
    YY=np.concatenate((np.arange(0,ty/2+1),np.arange(-ty/2+1,0)))
    YY=(YY.reshape((ty,1)))@np.ones((1,tx))
    # C'est une gaussienne, dont la moyenne est choisie de sorte que
    # l'integrale soit la meme que celle du filtre passe bas
    # (2*pi*sig^2=1/4*x*y (on a suppose que tx=ty))
    sig=(tx*ty)**0.5/2/(pi**0.5)
    mask=np.exp(-(XX**2+YY**2)/2/sig**2)
    imtf=np.fft.fft2(imt)*mask
    return np.real(np.fft.ifft2(imtf))

def get_gau_ker(s):
    ss=int(max(3,2*np.round(2.5*s)+1))
    ms=(ss-1)//2
    X=np.arange(-ms,ms+0.99)
    y=np.exp(-X**2/2/s**2)
    out=y.reshape((ss,1))@y.reshape((1,ss))
    out=out/out.sum()
    return out

def filtre_lineaire(im,mask):
    """ renvoie la convolution de l'image avec le mask. Le calcul se fait en 
utilisant la transformee de Fourier et est donc circulaire.  Fonctionne seulement pour 
les images en niveau de gris.
"""
    fft2=np.fft.fft2
    ifft2=np.fft.ifft2
    (y,x)=im.shape
    (ym,xm)=mask.shape
    mm=np.zeros((y,x))
    mm[:ym,:xm]=mask
    fout=(fft2(im)*fft2(mm))
    # on fait une translation pour ne pas avoir de decalage de l'image
    # pour un mask de taille impair ce sera parfait, sinon, il y a toujours un decalage de 1/2
    mm[:ym,:xm]=0
    y2=int(np.round(ym/2-0.5))
    x2=int(np.round(xm/2-0.5))
    mm[y2,x2]=1
    out=np.real(ifft2(fout*np.conj(fft2(mm))))
    return out


# %%
ima=io.imread('pyramide.tif')
#ima=filters.gaussian(ima,sigma)
ima = filtergauss(ima)
Alphas = [0.5]

for Alpha in Alphas:
    alpha= Alpha

    gradx=mr.dericheGradX(mr.dericheSmoothY(ima,alpha),alpha)
    grady=mr.dericheGradY(mr.dericheSmoothX(ima,alpha),alpha)  

    gradx2=mr.dericheGradX(mr.dericheSmoothY(gradx,alpha),alpha)
    grady2=mr.dericheGradY(mr.dericheSmoothX(grady,alpha),alpha) 
    
    lpima=gradx2+grady2
    posneg=(lpima>=0)
    nl,nc=ima.shape
    contours=np.uint8(np.zeros((nl,nc)))

    for i in range(1,nl):
        for j in range(1,nc):
            if (((i>0) and (posneg[i-1,j] != posneg[i,j])) or
                ((j>0) and (posneg[i,j-1] != posneg[i,j]))):
                contours[i,j]=255

    

    plt.figure(figsize=(8, 4))
    plt.figure('Contours')
    plt.imshow(contours, cmap='gray')
    plt.title(f"Alpha = ({alpha})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
  
# %%
