# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 21/05/2021
@version: 1.75
@Recommandation: Python 3.7
@But: KMedoids : déterminer le meilleur k
"""
import numpy as np
import sys

import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from collections import Counter
import matplotlib.pylab as plt


sys.path.insert(1,'../libs')

hemi='L'
d=np.load('../variables/'+hemi+'/matrix.npy')
index=np.load('../variables/'+hemi+'/matrix_index.npy',allow_pickle=True)
columns=np.load('../variables/'+hemi+'/matrix_columns.npy',allow_pickle=True)
data=pd.DataFrame((d>0.01)*d,index=index,columns=columns)#0 sur la diagonale, probleme de virgule flotante résolue

itermax=1000
cluster=list(range(2,30))
labels=[]
scores=[]
arg_scores=[]

for i in cluster:
    label=[]
    score=[]
    for j in range(itermax):
        kmedoids=KMedoids(n_clusters=i,metric='precomputed',init='k-medoids++',max_iter=1000).fit(data)
        label.append(kmedoids.labels_)#label pour un cluster donnée
        score.append(silhouette_score(data,kmedoids.labels_,metric='precomputed'))
        
    median=np.argsort(score)[len(score)//2]
    labels.append(label[median])#labels de l'ensemble des tirages pour un cluster 
    scores.append(score[median])

score=[i/itermax for i in score]
    
plt.figure()
plt.plot(cluster, scores, 'r+')
plt.title('Silhouette evolution according to the number of clusters')
plt.show()

if True:
    clus=12
    l=labels[clus]
    df=data.copy()
    columns = [df.columns.tolist()[i] for i in list(np.argsort(l))]
    df = df.reindex(columns, axis='columns')
    df = df.reindex(columns, axis='index')
    
    plt.figure()
    plt.imshow(df)
    plt.title('Nombre de cluster :'+str(cluster[clus-2]))
    plt.show()
