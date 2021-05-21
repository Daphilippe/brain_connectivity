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

itermax=10
cluster=list(range(2,100))
label=[]
score=[]
for j in range(10):
    for i in cluster:
            kmedoids=KMedoids(n_clusters=i  ,metric='precomputed',init='k-medoids++').fit(data)
            if j!=0:# first iteration
                label.append(kmedoids.labels_)
                score.append(silhouette_score(data,label[-1],metric='precomputed'))
            else:
                label[i-2]=label[i-2]+kmedoids.labels_# faire des paliers en fonction du nombre de cluster
                score[i-2]=score[i-2]+silhouette_score(data,kmedoids.labels_,metric='precomputed')
            
            if False:
                df=data.copy()
                columns = [df.columns.tolist()[i] for i in list(np.argsort(label))]
                df = df.reindex(columns, axis='columns')
                df = df.reindex(columns, axis='index')
                
                plt.figure()
                plt.imshow(df)
                plt.show()
score=[i/itermax for i in score]
for i in list(zip(cluster,label)):
    print(i)
    
plt.figure()
plt.plot(cluster, score, 'ro')
plt.title('Silhouette evolution according to the number of clusters')
plt.show()