# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 03/06/21
@version: 1.00
@Recommandation: Python 3.7
@revision 04/06/21
@But : Voisins Kmedoids sur isomap
"""
import numpy as np
import sys
sys.path.insert(1,'../libs')

import pandas as pd
from  sklearn.manifold import Isomap
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

if True:# Isomap 1 dimension, 2 voisins
    Lv= range(2,25)
    fig,axs=plt.subplots(len(Lv),sharex=True, sharey=True, figsize=(10,10))
    fig.subplots_adjust(hspace = 2 )
    fig.suptitle('Comparison for different values of neighbours')
    for v in Lv:
        iso=Isomap(n_neighbors=v,n_components=1,metric='precomputed')
        X_transformed=iso.fit_transform(data)  
        axs[v-2].yaxis.set_visible(False)
        axs[v-2].set_title(str(v)+' neighbours', pad=5)
        axs[v-2].scatter([i[0] for i in X_transformed],np.zeros(len(X_transformed)),marker='*')
    
    tools.save_fig(v,source2+'/isomap/dim1/') 
if True:# Isomap 2 dimension    
    Lv= range(0,25)
    fig,axs=plt.subplots(int(np.sqrt(len(Lv))),int(np.sqrt(len(Lv))),sharex=True, figsize=(10,10))
    fig.suptitle('Comparison for different values of neighbours')
    for v in Lv:
        iso=Isomap(n_neighbors=v+2,n_components=2,metric='precomputed')
        X_transformed=iso.fit_transform(data) 
          
        axs[int(np.floor(v/5)),int(v%5)].set_title(str(v+2)+' neighbours')
        axs[int(np.floor(v/5)),int(v%5)].scatter([i[0] for i in X_transformed],[i[1] for i in X_transformed],marker='*')          
    tools.save_fig(v,source2+'/isomap/dim2/')