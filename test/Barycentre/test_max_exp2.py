# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 14/05/2021
@version: 2.00
@Recommandation: Python 3.7
@Révision : 11/06/21
@But: comparaison barycentre avancées
- Permet de comparer le barycentre avec le sujet moyen en comparant la position des maximums locaux
"""
import numpy as np
import sys
import glob
from skimage.feature import peak_local_max
import matplotlib.pylab as plt


sys.path.insert(1,'../libs')
import tools, display

hemi='L'
source1='../../barycentre/'+hemi+'/'
source2='../../data/'+hemi+'/'
variables='../../variables/'+hemi+'/'
destination='../../variables/barycentre/local_max/'+hemi
size=9# a adapter si nécessaire dépend de l'origne des données

# Changement des données
i=0
min_distance=1
percent=90#seuillage des valeurs
grid_size=101
if False:
    dataX=None
    for np_name in glob.glob(str(source2)+'*.np[yz]'):
        X=np.load(np_name)
        _,_,data=tools.estimate_pseudo_density(X,grid_size)
        if dataX is None:
            dataX=data
        else:
            dataX=dataX+data
    dataX=dataX/np.max(dataX)
    tools.save_value(dataX,'mean',destination)
if True:
    data=np.load(destination+'/mean'+'.npy')    
    coord = peak_local_max((data>np.percentile(data, percent))*data, min_distance)
    tools.save_value(coord,'mean_coord',destination)
    
    plt.figure()
    extent = (0,grid_size-1 , 0,grid_size-1)
    plt.imshow(data,cmap=plt.cm.magma_r,origin='lower',extent=extent)
    plt.autoscale(False)
    plt.plot(coord[:, 1], coord[:, 0], 'g.')
    plt.axis('on')
    plt.xlabel('Precentral gyral crest scaled to 100')
    plt.ylabel('Postcentral gyral crest scaled to 100')
    plt.colorbar()
    plt.grid(linestyle = '--', linewidth = 0.5,alpha=0.5, which='major')
    plt.title(hemi+' hemisphere')
    tools.save_fig('mean',destination)