# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 11/05/2021
@version: 1.75
@revision: 11/06/21
@Recommandation: Python 3.7
"""
import numpy as np
import sys
from skimage.feature import peak_local_max
import matplotlib.pylab as plt

sys.path.insert(1,'../libs')
import tools

    
# Chemins
hemi='L'
source='../../data/'+hemi+'/'
variables='../../variables/'+hemi+'/'
destination='maximum/'+hemi+'/'

X=np.load('test/'+hemi+'/mean.npy')
min_distance=1
percent=90#seuillage des valeurs
coord = peak_local_max((X>np.percentile(X, percent))*X, min_distance)

plt.figure()
extent = (0, 100, 0, 100)
plt.imshow(X,cmap=plt.cm.magma_r,origin='lower',extent=extent)
plt.autoscale(False)
plt.plot(coord[:, 1], coord[:, 0], 'g.')
plt.axis('on')
plt.xlabel('Precentral gyral crest scaled to 100')
plt.ylabel('Post central gyral crest scaled to 100')
plt.colorbar()
plt.grid(linestyle = '--', linewidth = 0.5,alpha=0.5, which='major')
plt.title('Peak local max mean - '+hemi)
plt.show() 

print(coord)
####
grid_size=101# valeur par défaut
X=np.load('test/'+hemi+'/barycentre.npy')
_,_,data=tools.estimate_pseudo_density(X,grid_size)
data=data/np.max(data)
coord = peak_local_max((data>np.percentile(data, percent))*data, min_distance)

plt.figure()
extent = (0,grid_size-1 , 0,grid_size-1)
plt.imshow(data,cmap=plt.cm.magma_r,origin='lower',extent=extent)
plt.autoscale(False)
plt.plot(coord[:, 1], coord[:, 0], 'g.')
plt.axis('on')
plt.xlabel('Precentral gyral crest scaled to 100')
plt.ylabel('Post central gyral crest scaled to 100')
plt.colorbar()
plt.grid(linestyle = '--', linewidth = 0.5,alpha=0.5, which='major')
plt.title('Peak local max barycenter - '+hemi)
plt.show() 
print(coord)