# -*- coding: utf-8 -*-
"""
Created on Thu May 27 22:21:02 2021

@author: Daphilippe
"""

import numpy as np

import sys
import pandas as pd
from sklearn import preprocessing

sys.path.insert(1,'../libs')
import tools, display, barycenter


hemi='L'
source='../data/'+hemi+'/'
source2="../variables/clustering/"+hemi+"/"

columns=np.load('../variables/'+hemi+'/matrix_columns.npy',allow_pickle=True)
index=np.load('../variables/'+hemi+'/matrix_index.npy',allow_pickle=True)

if not(all(columns==index)):#controle intégrité matrice distance
    print('erreur')
    
cluster=np.load(source2+'cluster.npy')
labels=np.load(source2+'labels.npy')
k=labels[1] #labels[1]


c0=[]
w0=[]
c1=[]
w1=[]
c2=[]
w2=[]
for i in range(len(k)):
    if k[i]==0:
        c0.append(np.load(source+index[i]+'_connectivity_withHKnob.npy'))
        w0.append(  np.ones(np.shape(c0[-1])[0])/np.shape(c0[-1])[0] )
    if k[i]==1:
        c1.append(np.load(source+index[i]+'_connectivity_withHKnob.npy'))
        w1.append(  np.ones(np.shape(c1[-1])[0])/np.shape(c1[-1])[0] )
    if k[i]==2:
        c2.append(np.load(source+index[i]+'_connectivity_withHKnob.npy'))
        w2.append(  np.ones(np.shape(c2[-1])[0])/np.shape(c2[-1])[0] )
        
X_init = np.random.normal(0., 1., (2498, 2))#2087 pour R nombre de point moyen de l'ensemble des profils     
b=np.ones(np.shape(X_init)[0])/np.shape(X_init)[0]

X0=barycenter.free_support_barycenter(c0,w0,X_init,b)
tools.save_value(X0, 'X0',source2+'barycentre_k3')
X1=barycenter.free_support_barycenter(c1,w1,X_init,b)
tools.save_value(X1, 'X1',source2+'barycentre_k3')
X2=barycenter.free_support_barycenter(c2,w2,X_init,b)
tools.save_value(X2, 'X2',source2+'barycentre_k3')