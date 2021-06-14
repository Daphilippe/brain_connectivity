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

if True:# Isomap 1 dimension, 2 voisins
    bary=[]
    iso=Isomap(n_neighbors=20,n_components=1,metric='precomputed')
    X_transformed=iso.fit_transform(data) 
    trie=np.argsort(X_transformed,0)# sujets classés selon l'axe de projection
    
    #barycentre
    ### Paramètres du barycentre
    k=500
    X_init = np.random.normal(0., 1., (k, 2))  # initial Dirac locations
    b=np.ones(k)*1/k
    X=None
    
    ### Paramètre pour l'intervalle du barycentre glissant
    inter=10
    index_bis=index[trie]
    
    for i in range(len(index_bis)-inter):
        measures_locations=[]
        measures_weights=[]
        for j in index_bis[i:i+inter]:
            measures_locations.append(np.load(source+str(j[0])+'_connectivity_withHKnob.npy'))
            measures_weights.append(np.ones( np.shape(measures_locations[-1])[0] )*1/np.shape(measures_locations[-1])[0])
        bary.append(barycenter.iterative_barycenter(X,X_init,b,measures_locations,measures_weights,Nmax=inter,save=False,destination='/temp'))   
    tools.save_value(bary,'bary_glissant_'+str(inter)+'_'+hemi,'temp')    