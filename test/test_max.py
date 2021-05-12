# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 11/05/2021
@version: 1.5
@revision: 11/05/21
@Recommandation: Python 3.7
"""
import numpy as np
import sys
from skimage.feature import peak_local_max
import matplotlib.pylab as plt

sys.path.insert(1,'../libs')
import tools

    
# Chemins
source='../data/L/'
variables='../variables/L/'
destination='max'

X=np.load('test/L/mean.npy')
min_distance=1
percent=95
coord = peak_local_max((X>np.percentile(X, percent))*X, min_distance)

fig=plt.figure()
extent = (0, 100, 0, 100)
plt.imshow(X,cmap=plt.cm.magma_r,origin='lower',extent=extent)
plt.autoscale(False)
plt.plot(coord[:, 1], coord[:, 0], 'g.')
plt.axis('off')
plt.title('Peak local max mean')
####

X=np.load('test/L/barycentre.npy')
_,_,data=tools.estimate_pseudo_density(X)
data=data/np.max(data)
coord = peak_local_max((data>np.percentile(data, percent))*data, min_distance)

fig=plt.figure()
extent = (0, 100, 0, 100)
plt.imshow(data,cmap=plt.cm.magma_r,origin='lower',extent=extent)
plt.autoscale(False)
plt.plot(coord[:, 1], coord[:, 0], 'g.')
plt.axis('off')
plt.title('Peak local max barycenter')
