# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 03/06/21
@version: 1.00
@Recommandation: Python 3.7
@revision 
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

clus=2
if False:# Isomap 1 dimension, 2 voisins
    for cluster in np.load(source2+'/labels.npy'):
        iso=Isomap(n_neighbors=5,n_components=1,metric='precomputed')
        X_transformed=iso.fit_transform(data) 
        
        plt.figure(figsize=(20,20))
        plt.scatter([i[0] for i in X_transformed],[i[0] for i in X_transformed],marker='*',c=cluster)
        j=0
        for i in list(zip(index,X_transformed)):
            j=j+0.1
            plt.text(i[1]+j,i[1],i[0],fontsize=8)
            plt.title('Isomap dim 1 - '+str(clus)+' clusters')
        tools.save_fig(clus,source2+'/isomap_dim1_5vois/')
        clus=clus+1
clus=2    
if False:# Isomap 2 dimension    
    for cluster in np.load(source2+'/labels.npy'):
        iso=Isomap(n_neighbors=5,n_components=2,metric='precomputed')
        X_transformed=iso.fit_transform(data) 
        
        plt.figure(figsize=(10,10))
        plt.scatter([i[0] for i in X_transformed],[i[1] for i in X_transformed],marker='+',c=cluster)
        for i in list(zip(index,X_transformed)):
            plt.annotate(i[0], i[1],fontsize=8)
            
        plt.title('Isomap dim 2 - '+str(clus)+' clusters')
        tools.save_fig(clus,source2+'/isomap_dim2_5vois/')
        clus=clus+1

if True:# Isomap 3 dimension, 2 voisins
    for cluster in np.load(source2+'/labels.npy'):
        iso=Isomap(n_neighbors=2,n_components=3,metric='precomputed')
        X_transformed=iso.fit_transform(data) 
        
        fig=plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        ax.scatter3D([i[0] for i in X_transformed],[i[1] for i in X_transformed],[i[2] for i in X_transformed],marker='+',c=cluster)
        if False:
            for i in list(zip(index,X_transformed)):
                ax.text(i[1][0],i[1][1],i[1][2],'%s' % (str(i[0])),fontsize=7)    
        plt.show()        
    
    