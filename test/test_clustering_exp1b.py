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
from collections import Counter
import matplotlib.pylab as plt


sys.path.insert(1,'../libs')

hemi='L'
d=np.load('../variables/'+hemi+'/matrix.npy')
index=np.load('../variables/'+hemi+'/matrix_index.npy',allow_pickle=True)
columns=np.load('../variables/'+hemi+'/matrix_columns.npy',allow_pickle=True)
data=pd.DataFrame((d>0.01)*d,index=index,columns=columns)#0 sur la diagonale, probleme de virgule flotante résolue

itermax=20
cluster=list(range(2,np.shape(d)[0]))
labels=[]
score=[]
for j in range(itermax):
    label=[]
    for i in cluster:
        kmedoids=KMedoids(n_clusters=i  ,metric='precomputed',init='k-medoids++',max_iter=10000).fit(data)
        label.append(kmedoids.labels_)#label pour un cluster donnée
        if j==0:# first iteration
            score.append(silhouette_score(data,kmedoids.labels_,metric='precomputed'))
        else:
            score[i-2]=score[i-2]+silhouette_score(data,kmedoids.labels_,metric='precomputed')
    labels.append(label)#labels de l'ensemble des clusters par tirage

score=[i/itermax for i in score]
label=[]
for i in cluster:  
    label.append([k[i-2] for k in labels])#label pour un cluster par tirage
    
plt.figure()
plt.plot(cluster, score, 'r+')
plt.title('Silhouette evolution according to the number of clusters')
plt.show()

if False:        
    #Calcul des labels par vote majoritaire à améliorer si permutation de label équivalent d'une série à l'autre
    Labels_majority=[] #label après vote majoritaire pour chaque cluster
    for i in label:
        majorityvote=[]
        for j in range(np.shape(d)[0]):
            majorityvote.append(Counter([k[j] for k in i]).most_common()[0][0])#label pour un cluster par tirage
        Labels_majority.append(majorityvote)


    for i in list(zip(cluster,Labels_majority)):
        c,label=i
        df=data.copy()
        columns = [df.columns.tolist()[i] for i in list(np.argsort(label))]
        df = df.reindex(columns, axis='columns')
        df = df.reindex(columns, axis='index')
        
        plt.figure()
        plt.imshow(df)
        plt.title('Nombre de cluster : '+str(c))
        plt.show()
        if c>30:
            break
if True:
    clus=12
    for l in label[clus-2]:
        df=data.copy()
        columns = [df.columns.tolist()[i] for i in list(np.argsort(l))]
        df = df.reindex(columns, axis='columns')
        df = df.reindex(columns, axis='index')
        
        plt.figure()
        plt.imshow(df)
        plt.title('Nombre de cluster :'+str(cluster[clus-2]))
        plt.show()
        break
