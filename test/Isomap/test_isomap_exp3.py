# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 06/07/21
@version: 1.00
@Recommandation: Python 3.7
@But : Etude du nombre de point
"""
import numpy as np
import sys
sys.path.insert(1,'../../libs')

import pandas as pd
from  sklearn.manifold import Isomap
import matplotlib.pylab as plt

import tools, display, barycenter, process

# Directory
hemi='L'
source='../../data/'+hemi+'/'
source2="../../variables/clustering/"+hemi
source3='../../variables/'

columns=np.load(source3+hemi+'/matrix_columns.npy',allow_pickle=True)
index=np.load(source3+hemi+'/matrix_index.npy',allow_pickle=True)
if not(all(columns==index)):#controle intégrité matrice distance
    print('erreur')
    
d=np.load(source3+hemi+'/matrix.npy')
data=pd.DataFrame((d>0.01)*np.sqrt(d),index=index,columns=columns)

iso=Isomap(n_neighbors=7,n_components=1,metric='precomputed')
X_transformed=iso.fit_transform(data) 
cluster=np.load(source2+'/labels.npy')[0]


color=[np.shape(np.load(source+str(index[i][0])+'_connectivity_withHKnob.npy'))[0] for i in np.argsort(X_transformed,0)]
controle=[index[i][0] for i in np.argsort(X_transformed,0)]

fig,axs=plt.subplots(2,sharex=True, sharey=True, figsize=(10,10))
fig.subplots_adjust(hspace = 2 )
iso=Isomap(n_neighbors=7,n_components=1,metric='precomputed')
X_transformed=iso.fit_transform(data)  
axs[0].yaxis.set_visible(False)
axs[0].set_title(hemi+' hemisphere', pad=5)
axs[0].scatter([i[0] for i in X_transformed],np.ones(len(X_transformed))*np.mean(color),marker='*')

axs[1].yaxis.set_visible(True)
axs[1].set_title('Number of dots', pad=5)
axs[1].scatter([i[0] for i in X_transformed],color,marker='*',c=cluster)