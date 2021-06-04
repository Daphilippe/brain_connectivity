# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 04/06/21
@version: 1.00
@Recommandation: Python 3.7
@revision 04/06/21
@But : influence nb voisin sur isomap (peu se faire sous la forme d'un dictionnaire)
"""
import numpy as np
import sys
sys.path.insert(1,'../libs')

import pandas as pd
from  sklearn.manifold import Isomap

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

clus=6
cluster= np.load(source2+'/labels.npy')[clus-2]

# corrélation intrinsèque
if False:# Isomap 1 dimension, 2 voisins
    temp=[]
    voisins=range(2,25)
    for k in voisins:
        iso=Isomap(n_neighbors=k,n_components=1,metric='precomputed')
        X_transformed=iso.fit_transform(data)   
        [a]=np.argsort(X_transformed,0).T
        temp.append(a.T)     
    print(np.sum(np.sum(pd.DataFrame(np.array(temp)).T.corr())/len(voisins))/len(voisins))#corrélation moyenne entre isomap 1d


# corrélation par cluster
if True:# 
    temp=[]
    voisins=range(2,25)
    axis=0
    for k in voisins:
        iso=Isomap(n_neighbors=k,n_components=3,metric='precomputed')
        X_transformed=iso.fit_transform(data)  
        
        d={}
        for i in range(clus):
            d[i]=[]
        for i in enumerate(np.argsort(X_transformed,0)):
            d[cluster[i[0]]].append(i[1][axis])# on observe selon une composante
        for i in d:
            d[i]=np.sort(d[i])
        temp.append(d)
    a=pd.DataFrame.from_dict(temp)
    
    for i in a:
        temp=[]
        for j in a[i]:
            temp.append(j)
        print(np.min(np.min(pd.DataFrame(temp).T.corr())))# la plus petite corrélation