# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 21/07/21
@version: 1.00
@Recommandation: Python 3.7
@But : DBSCAN continu
"""
import numpy as np
import sys
sys.path.insert(1,'../../libs')

from sklearn.cluster import DBSCAN
from sklearn import metrics

import matplotlib.pylab as plt

import tools, display, barycenter, process

# Directory
hemi='R'
source='../../data/'+hemi+'/'
source1='../../variables/'
chemin=source1+'DBSCAN/'

# Importation données
barycentre=np.load(source1+"barycentre/"+hemi+'/99.npy')
fenetre=31
Lglissant=np.load(source1+'isomap/'+hemi+'/barycentre glissant/bary_glissant_'+str(fenetre)+'_'+hemi+'.npy')


X=barycentre
db = DBSCAN(eps=2.25, min_samples=int(np.shape(X)[0]*0.01)).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

_,_,img_xs=tools.estimate_pseudo_density(X)
img_xs=img_xs/np.sum(img_xs)#densité = 1
image=(img_xs>np.percentile(img_xs,90))*img_xs

# #############################################################################
# Plot result
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
unique_labels.remove(-1)
plt.figure()
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = None

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)
    
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], '*', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)
    
plt.imshow(image,cmap=plt.cm.magma_r,origin='lower',extent=(0, 100, 0, 100))  
plt.title(hemi + ' - estimated number of clusters: %d' % n_clusters_)
plt.show()