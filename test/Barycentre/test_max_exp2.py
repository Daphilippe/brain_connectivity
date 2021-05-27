# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 14/05/2021
@version: 1.25
@Recommandation: Python 3.7
@Révision : 28/05/21
@But: comparaison barycentre avancées
"""
import numpy as np
import sys
import glob
from skimage.feature import peak_local_max
import matplotlib.pylab as plt


sys.path.insert(1,'../libs')
import tools, display

source='barycentre/R/'
variables='../variables/R/'
destination='../variables/local_max/'
size=len(source)

# Changement des données
i=0
min_distance=1
percent=90#seuillage des valeurs
grid_size=101

if False:
    for np_name in glob.glob(str(source)+'*.np[yz]'):
        X=np.load(np_name)
        _,_,data=tools.estimate_pseudo_density(X,grid_size)
        data=data/np.max(data)
        
        coord = peak_local_max((data>np.percentile(data, percent))*data, min_distance)
        tools.save_value(coord,'coord_'+str(np_name[size:]),destination)
        
        plt.figure()
        extent = (0,grid_size-1 , 0,grid_size-1)
        plt.imshow(data,cmap=plt.cm.magma_r,origin='lower',extent=extent)
        plt.autoscale(False)
        plt.plot(coord[:, 1], coord[:, 0], 'g.')
        plt.axis('off')
        plt.title('Peak local max barycenter')
        tools.save_fig(str(np_name[size:]),destination)
if False:
    dataX=None
    source='../data/R/'
    destination='local_max/'
    for np_name in glob.glob(str(source)+'*.np[yz]'):
        X=np.load(np_name)
        _,_,data=tools.estimate_pseudo_density(X,grid_size)
        if dataX is None:
            dataX=data
        else:
            dataX=dataX+data
    dataX=dataX/np.max(dataX)
    tools.save_value(dataX,'mean_R',destination)
if True:
    destination='../variables/local_max/'    
    data=np.load(destination+'mean_R.npy')    
    coord = peak_local_max((data>np.percentile(data, percent))*data, min_distance)
    tools.save_value(coord,'coord_R',destination)
    
    plt.figure()
    extent = (0,grid_size-1 , 0,grid_size-1)
    plt.imshow(data,cmap=plt.cm.magma_r,origin='lower',extent=extent)
    plt.autoscale(False)
    plt.plot(coord[:, 1], coord[:, 0], 'g.')
    plt.axis('off')
    plt.title('Right hemisphere')
    tools.save_fig('mean_R',destination)