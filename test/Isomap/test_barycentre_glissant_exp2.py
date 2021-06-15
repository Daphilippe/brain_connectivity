# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 11/06/2021
@version: 1.20
@Recommandation: Python 3.7
@révision:14/06/21
@But : Barycentre glissant
"""
import numpy as np
import sys
sys.path.insert(1,'../../libs')

import pandas as pd
from  sklearn.manifold import Isomap
from skimage.feature import peak_local_max
import matplotlib.pylab as plt

import tools, display, barycenter, process

# Directory
hemi='R'
source='../../data/'+hemi+'/'
source2='../../variables/isomap/'+hemi+'/barycentre glissant/'
source3="../../variables/"

columns=np.load(source3+hemi+'/matrix_columns.npy',allow_pickle=True)
index=np.load(source3+hemi+'/matrix_index.npy',allow_pickle=True)
if not(all(columns==index)):#controle intégrité matrice distance
    print('erreur')
    
d=np.load(source3+hemi+'/matrix.npy')
data=pd.DataFrame((d>0.01)*np.sqrt(d),index=index,columns=columns)

inter=31
bary=np.load(source2+'bary_glissant_'+str(inter)+'_'+hemi+'.npy')

try:
    trie=np.load(source2+'isomap_index_'+hemi+'.npy')
except:
    iso=Isomap(n_neighbors=10,n_components=1,metric='precomputed')
    X_transformed=iso.fit_transform(data) 
    trie=np.argsort(X_transformed,0)


min_distance=1
percent=90#seuillage des valeurs
grid_size=101# valeur par défaut
liste=[]
if True:
    for i in enumerate(bary):
        text='Barycenter of:'
        for j in index[trie[i[0]:i[0]+inter]]:
            text+=' '+str(j[0])
        _,_,img_xs=tools.estimate_pseudo_density(i[1])
        #data=img_xs/np.max(img_xs)
        liste.append(img_xs)
liste=liste/np.max(liste)
if True:
    for i in enumerate(liste):
            coord = peak_local_max((i[1]>np.percentile(i[1], percent))*i[1], min_distance)
            plt.figure()
            extent = (0,grid_size-1 , 0,grid_size-1)
            plt.imshow(i[1],cmap=plt.cm.magma_r,origin='lower',extent=extent)
            plt.autoscale(False)
            plt.plot(coord[:, 1], coord[:, 0], 'g.')
            plt.axis('on')
            plt.xlabel('Precentral gyral crest scaled to 100')
            plt.ylabel('Post central gyral crest scaled to 100')
            plt.colorbar()
            plt.clim(0,1)
            plt.grid(linestyle = '--', linewidth = 0.5,alpha=0.5, which='major')
            plt.title('Central subject n°'+str(i[0]+int(inter/2)))
            tools.save_fig(str(i[0]),source2+str(inter)+'/')
            #tools.save_value(coord, str(i[0]),source2+'/'+str(inter)+'/')