# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 20/04/2021
@version: 1.75
@Recommandation: Python 3.7
@revision: 21/04/2021
@But: clustering
"""
import numpy as np

import sys
import pandas as pd
import matplotlib.pylab as plt

hemi='L'
d=np.load('../variables/'+hemi+'/matrix.npy')
index=np.load('../variables/'+hemi+'/matrix_index.npy',allow_pickle=True)
columns=np.load('../variables/'+hemi+'/matrix_columns.npy',allow_pickle=True)
data=pd.DataFrame((d>0.01)*d,index=index,columns=columns)#0 sur la diagonale, probleme de virgule flotante résolue
if False:
    plt.figure()
    plt.imshow(d)
    plt.show()
# diagonalisation
if False: #analyse spectrale
    from numpy import linalg as LA
    df=data.copy()
    dfb= 1-df/np.max(df)# convert distance matrix to similarity matrix
    q, eig_v = LA.eig(np.sqrt(dfb))# on a np.dot(np.dot(v,np.diag(w)),v.transpose())=d
    # trie croisant
    eigv_or=np.fliplr(eig_v)
    q_or=q[::-1]
    
    # vecteur v
    v=eigv_or.T[1]
    # calcul
    v2=np.dot(np.diag(q_or/np.sum(q_or)),v)
    # récupérer les positions lors du trie croissant
    p=99-np.argsort(v2)#trie décroissant
    
    columns = [df.columns.tolist()[i] for i in p]
    df = df.reindex(columns, axis='columns')
    df = df.reindex(columns, axis='index')
    
    plt.figure()
    plt.imshow(df)
    plt.show()
    print(p)

if True:# clustering hierarchique
    import scipy.cluster.hierarchy as sch
    df=data.copy()
    dist=sch.ward(np.sqrt(df))
    
    cluster=100 # 3 clusters principauxs #np.shape(d)[0]
    #label = sch.fcluster(dist,cluster,criterion='distance') # or distance avec un seuille de différence
    label=sch.fcluster(dist,cluster,criterion='maxclust')
    plt.figure()
    sch.dendrogram(dist)
    plt.show()
    
    columns = [df.columns.tolist()[i] for i in list(np.argsort(label))]
    df = df.reindex(columns, axis='columns')
    df = df.reindex(columns, axis='index')
    
    plt.figure()
    plt.imshow(df)
    plt.show()
    
    print(np.argsort(label))
sys.exit()