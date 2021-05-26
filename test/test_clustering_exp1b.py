# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 21/05/2021
@version: 1.75
@Recommandation: Python 3.7
@revision: 26/05/2021
@But: KMedoids : déterminer le meilleur k
"""
import numpy as np
import sys

import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pylab as plt

sys.path.insert(1,'../libs')
import tools,display


sys.path.insert(1,'../libs')
chemin="../variables/clustering/"
hemi='L'
d=np.load('../variables/'+hemi+'/matrix.npy')
index=np.load('../variables/'+hemi+'/matrix_index.npy',allow_pickle=True)
columns=np.load('../variables/'+hemi+'/matrix_columns.npy',allow_pickle=True)
data=pd.DataFrame((d>0.01)*np.sqrt(d),index=index,columns=columns)#0 sur la diagonale, probleme de virgule flotante résolue

if False:
    # détermination de elbows et silhouette score + labels + lissage en prenant la valeur median
    itermax=2000
    cluster=list(range(2,100))
    labels=[]
    scores=[]
    scores_bis=[]
    arg_scores=[]
    
    for i in cluster:# pour chaque cluster
        label=[]
        score=[]
        score_bis=[]
        for j in range(itermax):# on répète l'expérience
            kmedoids=KMedoids(n_clusters=i,metric='precomputed',init='k-medoids++',max_iter=1000).fit(data)
            label.append(kmedoids.labels_)#label pour un cluster donnée
            score.append(silhouette_score(data,kmedoids.labels_,metric='precomputed'))
            score_bis.append(kmedoids.inertia_)
        
        # sauvegarde du label représentatif
        median=np.argsort(score)[len(score)//2]
        labels.append(label[median])#labels de l'ensemble des tirages pour un cluster 
        scores.append(score[median])
        scores_bis.append(score_bis[median])
        
        if False:#mean au lieu de mediane
            mean=np.mean(score)
            scores.append(mean)
            scores_bis.append(np.mean(score_bis))
    
    score=[i/itermax for i in score]
    
    #affichage    
    plt.figure()
    plt.plot(cluster, scores, 'ro')
    plt.title('Silhouette evolution according to the number of clusters')
    plt.show()
    
    plt.figure()
    plt.plot(cluster, scores_bis, 'ro')
    plt.title('Elbow score')
    plt.show()
    
    if False:#sauvegarde des données
        tools.save_value(value=cluster, title='cluster',directory=chemin)
        tools.save_value(value=scores, title='scores',directory=chemin)
        tools.save_value(value=scores_bis, title='scores_bis',directory=chemin)
        tools.save_value(value=labels, title='labels',directory=chemin)

if True:
    # chargement des données déjà générées
    cluster=np.load('../variables/clustering/cluster.npy')
    scores=np.load('../variables/clustering/scores.npy')
    scores_bis=np.load('../variables/clustering/scores_bis.npy')
    labels=np.load('../variables/clustering/labels.npy')
    
    # affichage
    plt.figure()
    plt.plot(cluster[:30], scores[:30], 'r+')
    plt.title('Silhouette evolution according to the number of clusters')
    plt.show()
    
    plt.figure()
    plt.plot(cluster[:30], [np.sqrt(i) for i in scores_bis[:30]], 'r+')
    plt.title('Elbow score')
    plt.show()

    # réorganisation 
    clus=2
    l=labels[clus-2]
    df=data.copy()
    columns = [df.columns.tolist()[i] for i in list(np.argsort(l))]
    df = df.reindex(columns, axis='columns')
    df = df.reindex(columns, axis='index')
    
    display.show_map(df**2,'Number of cluster :'+str(cluster[clus-2]),cmap=None,origin='upper')
