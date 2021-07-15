# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 08/07/21
@version: 1.00
@Recommandation: Python 3.7
@But : Isomap - Comparaison des deux hémisphères
"""
import numpy as np
import sys
sys.path.insert(1,'../../libs')

import pandas as pd
from  sklearn.manifold import Isomap
import matplotlib.pylab as plt

import tools, display, barycenter, process

# Hémisphère gauche
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

# Calcul isomap
iso=Isomap(n_neighbors=7,n_components=1,metric='precomputed')
X_transformedL=iso.fit_transform(data) 
clusterL=np.load(source2+'/labels.npy')[0]

color=[i for i in X_transformedL]

# Hémisphère droit
# Directory
hemi='R'
source='../../data/'+hemi+'/'
source2="../../variables/clustering/"+hemi
source3='../../variables/'

columns=np.load(source3+hemi+'/matrix_columns.npy',allow_pickle=True)
index=np.load(source3+hemi+'/matrix_index.npy',allow_pickle=True)
if not(all(columns==index)):#controle intégrité matrice distance
    print('erreur')
    
d=np.load(source3+hemi+'/matrix.npy')
data=pd.DataFrame((d>0.01)*np.sqrt(d),index=index,columns=columns)

# Calcul isomap
iso=Isomap(n_neighbors=7,n_components=1,metric='precomputed')
X_transformedR=iso.fit_transform(data) 
clusterR=np.load(source2+'/labels.npy')[0]

fig,axs=plt.subplots(2,sharex=True, sharey=True, figsize=(10,10))
fig.subplots_adjust(hspace = 2 )
axs[0].yaxis.set_visible(False)
axs[0].set_title('L hemisphere', pad=5)
axs[0].scatter([i[0] for i in X_transformedL],np.zeros(len(X_transformedL)),marker='*',c=color)

axs[1].yaxis.set_visible(False)
axs[1].set_title('R hemisphere', pad=5)
axs[1].scatter([i[0] for i in X_transformedR],np.zeros(len(X_transformedR)),marker='*',c=color)