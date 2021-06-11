# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 11/06/2021
@version: 1.00
@Recommandation: Python 3.7
@But : Barycentre glissant
"""
import numpy as np
import sys
sys.path.insert(1,'../libs')

import pandas as pd
from  sklearn.manifold import Isomap
from skimage.feature import peak_local_max
import matplotlib.pylab as plt

import tools, display, barycenter, process

# Directory
hemi='R'
source='../data/'+hemi+'/'
source2="../variables/clustering/"+hemi

columns=np.load('../variables/'+hemi+'/matrix_columns.npy',allow_pickle=True)
index=np.load('../variables/'+hemi+'/matrix_index.npy',allow_pickle=True)
if not(all(columns==index)):#controle intégrité matrice distance
    print('erreur')
    
d=np.load('../variables/'+hemi+'/matrix.npy')
data=pd.DataFrame((d>0.01)*np.sqrt(d),index=index,columns=columns)

inter=10
bary=np.load('temp/bary_glissant_'+str(inter)+'_'+hemi+'.npy')

iso=Isomap(n_neighbors=20,n_components=1,metric='precomputed')
X_transformed=iso.fit_transform(data) 
trie=np.argsort(X_transformed,0)


min_distance=1
percent=90#seuillage des valeurs
grid_size=101# valeur par défaut

for i in enumerate(bary):
    text='Barycenter of:'
    for j in index[trie[i[0]:i[0]+inter]]:
        text+=' '+str(j[0])
    _,_,img_xs=tools.estimate_pseudo_density(i[1])
    data=img_xs/np.max(img_xs)
    
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
    plt.title('Step: '+str(i[0]))
    tools.save_fig(str(i[0]),'temp/'+hemi+'/'+str(inter)+'/')
    tools.save_value(coord, str(i[0]),'temp/'+hemi+'/'+str(inter)+'/')