# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 21/07/21
@version: 1.00
@Recommandation: Python 3.7
@But : Study of density
"""
import numpy as np
import sys
sys.path.insert(1,'../../libs')

from sklearn.cluster import DBSCAN
from sklearn import metrics

import matplotlib.pylab as plt

import tools, display, barycenter, process

# Directory
hemi='L'
source='../../data/'+hemi+'/'
source1='../../variables/'
chemin=source1+'DBSCAN/'

# Importation données
barycentre=np.load(source1+"barycentre/"+hemi+'/99.npy')
fenetre=31
Lglissant=np.load(source1+'isomap/'+hemi+'/barycentre glissant/bary_glissant_'+str(fenetre)+'_'+hemi+'.npy')

# Barycentre
X=barycentre
_,_,img_xs=tools.estimate_pseudo_density(X)
img_xs=img_xs/np.sum(img_xs)
image=(img_xs>np.percentile(img_xs,90))*img_xs
print('Densité pertinante restante : ',np.sum(image))
display.show_map(image, hemi+' - test')


print('Diagonale inférieur ventrale : ',np.sum(image[:40,:40]),' par rapport à la densité restante : ',np.sum(image[:40,:40])/np.sum(image))
print('Diagonale supérieur dorsale : ',np.sum(image[40:,40:]),' par rapport à la densité restante : ',np.sum(image[40:,40:])/np.sum(image))

# Suivi
Lventral=[]
Ldorsal=[]
for X in Lglissant:
    _,_,img_xs=tools.estimate_pseudo_density(X)
    img_xs=img_xs/np.sum(img_xs)
    image=(img_xs>np.percentile(img_xs,90))*img_xs
    Lventral.append(np.sum(image[:40,:40]))
    Ldorsal.append(np.sum(image[40:,40:]))