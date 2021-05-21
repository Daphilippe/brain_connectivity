# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 21/05/2021
@version: 1.00
@Recommandation: Python 3.7
@But: KMedoids : déterminer le meilleur k
"""
import numpy as np
import sys

import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

import matplotlib.pylab as plt


sys.path.insert(1,'../libs')
import tools,display,barycenter,process

hemi='L'
d=np.load('../variables/'+hemi+'/matrix.npy')
index=np.load('../variables/'+hemi+'/matrix_index.npy',allow_pickle=True)
columns=np.load('../variables/'+hemi+'/matrix_columns.npy',allow_pickle=True)
data=pd.DataFrame((d>0.01)*d,index=index,columns=columns)#0 sur la diagonale, probleme de virgule flotante résolue

itermax=3
cluster=list(range(2,np.shape(d)[0]))
labels=[]
score=[]
for j in range(itermax):
    label=[]
    for i in cluster:
        kmedoids=KMedoids(n_clusters=i  ,metric='precomputed',init='k-medoids++').fit(data)
        label.append(kmedoids.labels_)
        if j==0:# first iteration
            score.append(silhouette_score(data,kmedoids.labels_,metric='precomputed'))
        else:
            score[i-2]=score[i-2]+silhouette_score(data,kmedoids.labels_,metric='precomputed')
    labels.append(label)

score=[i/itermax for i in score]

label=[]
for i in cluster:  
    label.append([k[i-2] for k in labels]
    
plt.figure()
plt.plot(cluster, score, 'ro')
plt.title('Silhouette evolution according to the number of clusters')
plt.show()