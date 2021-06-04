# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 04/06/21
@version: 1.00
@Recommandation: Python 3.7
@revision 04/06/21
@But : influence nb voisin sur isomap
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
cluster= np.load(source2+'/labels.npy')[clus-2]

temp=[]
if True:# Isomap 1 dimension, 2 voisins
    for k in range(2,20):
        iso=Isomap(n_neighbors=k,n_components=1,metric='precomputed')
        X_transformed=iso.fit_transform(data)   
        [a]=np.argsort(X_transformed,0).T
        temp.append(a.T)     
a=pd.DataFrame(np.array(temp)).T.corr()
if True:# Isomap 2 dimension    
    for k in range(2,20):
        iso=Isomap(n_neighbors=15,n_components=2,metric='precomputed')
        X_transformed=iso.fit_transform(data) 
        

if False:# Isomap 3 dimension, 2 voisins
    for k in range(2,20):
        iso=Isomap(n_neighbors=15,n_components=3,metric='precomputed')
        X_transformed=iso.fit_transform(data) 
    