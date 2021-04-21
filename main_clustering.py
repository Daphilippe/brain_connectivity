# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 20/04/2021
@version: 1.75
@Recommandation: Python 3.7
@revision: 21/04/2021
@But: clustering
@Step: 2
"""
import numpy as np

import sys
import pandas as pd
import matplotlib.pylab as plt

# Left
d=np.load('./variables_L/matrix_L.npy')
index=np.load('./variables_L/matrix_L_index.npy',allow_pickle=True)
columns=np.load('./variables_L/matrix_L_columns.npy',allow_pickle=True)
data=pd.DataFrame((d>0.01)*d,index=index,columns=columns)#0 sur la diagonale, probleme de virgule flotante résolue

# diagonalisation
if True: #analyse spectrale
    from numpy import linalg as LA
    df=data.copy()
    q, eig_v = LA.eig(d)# on a np.dot(np.dot(v,np.diag(w)),v.transpose())=d
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
    sim = np.exp(- 0.5*(d/np.max(d))**2) # pour obtenir un similarity matrix
    L = sch.linkage(sim, method='weighted')#alias WPGMA --complete,centroid,average,weighted
    plt.figure()
    sch.dendrogram(L)
    plt.show()
    
    cluster=100#np.shape(d)[0]
    label = sch.fcluster(L,cluster,'maxclust') # or distance avec un seuille de différence
    columns = [df.columns.tolist()[i] for i in list(np.argsort(label))]
    df = df.reindex(columns, axis='columns')
    df = df.reindex(columns, axis='index')
    
    plt.figure()
    plt.imshow(df)
    plt.show()
    
    print(np.argsort(label))
sys.exit()