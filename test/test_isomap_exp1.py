# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 27/05/2021
@version: 1.25
@Recommandation: Python 3.7
@revision 03/06/21
@But : Afficher les données en respectant la géodésie des données
"""
import numpy as np
import sys
sys.path.insert(1,'../libs')

import pandas as pd
from  sklearn.manifold import Isomap
import matplotlib.pylab as plt

import tools, display, barycenter, process

# Directory
hemi='L'
source='../data/'+hemi+'/'
source2="../variables/clustering/"+hemi

columns=np.load('../variables/'+hemi+'/matrix_columns.npy',allow_pickle=True)
index=np.load('../variables/'+hemi+'/matrix_index.npy',allow_pickle=True)
if not(all(columns==index)):#controle intégrité matrice distance
    print('erreur')
    
d=np.load('../variables/'+hemi+'/matrix.npy')
data=pd.DataFrame((d>0.01)*np.sqrt(d),index=index,columns=columns)

clus=4
cluster=np.load(source2+'/labels.npy')[clus-2]#labels des différents clusters

if False:# Isomap 1 dimension, 4 voisins
    iso=Isomap(n_neighbors=4,n_components=1,metric='precomputed')
    X_transformed=iso.fit_transform(data) 
    
    plt.figure(figsize=(20,20))
    plt.scatter([i[0] for i in X_transformed],[i[0] for i in X_transformed],marker='*',c=cluster)
    j=0
    for i in list(zip(index,X_transformed)):
        j=j+0.1
        plt.text(i[1]+j,i[1],i[0],fontsize=8)
    #plt.legend()
    plt.show()
    
if False:# Isomap 2 dimension, 3 voisins
    iso=Isomap(n_neighbors=3,n_components=2,metric='precomputed')
    X_transformed=iso.fit_transform(data)
    
    plt.figure(figsize=(10,10))
    plt.scatter([i[0] for i in X_transformed],[i[1] for i in X_transformed],marker='+',c=cluster)
    for i in list(zip(index,X_transformed)):
        plt.annotate(i[0], i[1],fontsize=8)
    #plt.legend()
    plt.show()
    
    
from IPython.display import HTML
from matplotlib import animation
def animate(frame):
  ax.view_init(20, frame)
  plt.pause(.001)
  return frame

    
if True:# Isomap 3 dimension, 2 voisins
    iso=Isomap(n_neighbors=2,n_components=3,metric='precomputed')
    X_transformed=iso.fit_transform(data)
    
    fig=plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax.scatter3D([i[0] for i in X_transformed],[i[1] for i in X_transformed],[i[2] for i in X_transformed],marker='+',c=cluster)
    if False:
        for i in list(zip(index,X_transformed)):
            ax.text(i[1][0],i[1][1],i[1][2],'%s' % (str(i[0])),fontsize=7)    
    anim = animation.FuncAnimation(fig, animate, frames=120, interval=120)
    