# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 03/06/21
@version: 2.00
@Recommandation: Python 3.7
@revision 11/06/21
@But : Cluster Kmedoids sur isomap
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
source='../../data/'+hemi+'/'
source2="../../variables/clustering/"+hemi
source3='../../variables/'

columns=np.load(source3+hemi+'/matrix_columns.npy',allow_pickle=True)
index=np.load(source3+hemi+'/matrix_index.npy',allow_pickle=True)
if not(all(columns==index)):#controle intégrité matrice distance
    print('erreur')
    
d=np.load(source3+hemi+'/matrix.npy')
data=pd.DataFrame((d>0.01)*np.sqrt(d),index=index,columns=columns)

clus=2
if True:# Isomap 1 dimension, 2 voisins
    for cluster in np.load(source2+'/labels.npy'):
        iso=Isomap(n_neighbors=20,n_components=1,metric='precomputed')
        X_transformed=iso.fit_transform(data) 
        
        plt.figure(figsize=(10,3))
        plt.scatter([i[0] for i in X_transformed],np.zeros(len(X_transformed)),marker='*',c=cluster)
        plt.ylim(-1,1)
        j=0
        for i in np.argsort(X_transformed,0):
            j=j+1
            plt.text(X_transformed[i]-0.2,(np.mod(j,10)-5)/10,int(i),fontsize=10)
        tools.save_fig(clus,source2+'/isomap/dim1/')
        clus=clus+1
clus=2    
if False:# Isomap 2 dimension    
    for cluster in np.load(source2+'/labels.npy'):
        iso=Isomap(n_neighbors=20,n_components=2,metric='precomputed')
        X_transformed=iso.fit_transform(data) 
        
        plt.figure(figsize=(10,10))
        plt.scatter([i[0] for i in X_transformed],[i[1] for i in X_transformed],marker='+',c=cluster)
        for i in enumerate(X_transformed):
            plt.annotate(i[0],i[1],fontsize=10)
            
        plt.title('Isomap dim 2 - '+str(clus)+' clusters')
        tools.save_fig(clus,source2+'/isomap/dim2/')
        clus=clus+1

if False:# Isomap 3 dimension, 2 voisins
    for cluster in np.load(source2+'/labels.npy'):
        iso=Isomap(n_neighbors=15,n_components=3,metric='precomputed')
        X_transformed=iso.fit_transform(data) 
        
        fig=plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        ax.scatter3D([i[0] for i in X_transformed],[i[1] for i in X_transformed],[i[2] for i in X_transformed],marker='+',c=cluster)
        if True:
            for i in enumerate(X_transformed):
                ax.text(i[1][0],i[1][1],i[1][2],'%s' % (str(i[0])),fontsize=7)       
        plt.show()        
    
    