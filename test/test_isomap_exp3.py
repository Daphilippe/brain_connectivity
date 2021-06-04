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

clus=2
cluster= np.load(source2+'/labels.npy')[clus-2]
                
# corrélation par cluster
if True:# 
    voisins=range(2,25)
    n_components=3
    view=[]
    for k in voisins:# on change le nombre de voisin
        temp=[]
        for axis in range(n_components):
            iso=Isomap(n_neighbors=k,n_components=n_components,metric='precomputed')
            X_transformed=iso.fit_transform(data)  
            
            d={}
            for i in range(clus):# regroupement des points par cluster
                d[i]=[]
            for i in enumerate(np.argsort(X_transformed,0)):#récupération de l'ordre établit par l'isomap en fonction de l'axe
                d[cluster[i[0]]].append(i[1][axis])# on observe selon une composante
            for i in d:#trie par cluster de l'ordre des points dans l'isomap selon l'axe 
                d[i]=np.sort(d[i])
            temp.append(d)
        a=pd.DataFrame.from_dict(temp)
        
        # reconditionnement des données pour calcul de la corrélation par cluster
        temp2=[]
        for i in a:
            temp=[]
            for j in a[i]:
                temp.append(j)
            temp2.append(np.min(np.min(pd.DataFrame(temp).T.corr())))# la plus petite corrélation sur les n axes
        view.append(temp2)
    print(np.min(view))# on a la corrélation minimal sur notre jeu de donnée