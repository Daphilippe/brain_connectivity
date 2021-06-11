# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 21/05/2021
@version: 2.00
@Recommandation: Python 3.7
@revision: 31/05/2021
@But: KMedoids : déterminer le meilleur k
"""
import numpy as np
import sys

import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pylab as plt

sys.path.insert(1,'../../libs')
import tools,display


sys.path.insert(1,'../libs')
hemi='L'
chemin="../../variables/clustering/"+hemi
d=np.load('../../variables/'+hemi+'/matrix.npy')
index=np.load('../../variables/'+hemi+'/matrix_index.npy',allow_pickle=True)
columns=np.load('../../variables/'+hemi+'/matrix_columns.npy',allow_pickle=True)
data=pd.DataFrame((d>0.01)*np.sqrt(d),index=index,columns=columns)#0 sur la diagonale, probleme de virgule flotante résolue

itermax=5000
if False:
    # détermination de elbows et silhouette score + labels + lissage en prenant la valeur median
    cluster=list(range(2,10))
    labels=[]
    scores=[]
    scores_bis=[]
    
    for i in cluster:# pour chaque cluster
        score_bis=10E10
        for j in range(itermax):# on répète l'expérience
            kmedoids=KMedoids(n_clusters=i,metric='precomputed',init='k-medoids++',max_iter=10).fit(data)
            if score_bis>kmedoids.inertia_:
                score_bis=kmedoids.inertia_
                label=kmedoids.labels_#label pour un cluster donnée
                score=silhouette_score(data,kmedoids.labels_,metric='precomputed')
        labels.append(label)#labels de l'ensemble des tirages pour un cluster 
        scores.append(score)
        scores_bis.append(score_bis)
    
    if False:#sauvegarde des données
        tools.save_value(value=cluster, title='cluster',directory=chemin)
        tools.save_value(value=scores, title='scores',directory=chemin)
        tools.save_value(value=scores_bis, title='scores_bis',directory=chemin)
        tools.save_value(value=labels, title='labels',directory=chemin)

if True:
    # chargement des données déjà générées
    cluster=np.load(chemin+'/cluster.npy')
    scores=np.load(chemin+'/scores.npy')
    scores_bis=np.load(chemin+'/scores_bis.npy')
    labels=np.load(chemin+'/labels.npy')
    
    # affichage
    plt.figure()
    plt.plot(cluster[:30], scores[:30], 'r+')
    plt.title('Silhouette score - '+hemi)
    tools.save_fig('silhouette_'+hemi,chemin)
    
    plt.figure()
    plt.plot(cluster[:30], [np.sqrt(i) for i in scores_bis[:30]], 'r+')
    plt.title('Elbow score - '+hemi)
    tools.save_fig('elbow_'+hemi,chemin)

    # réorganisation
if True:
    clus=1
    for l in labels:
        clus=clus+1
        df=data.copy()
        columns = [df.columns.tolist()[i] for i in list(np.argsort(l))]
        df = df.reindex(columns, axis='columns')
        df = df.reindex(columns, axis='index')
        display.show_map(df**2,'Number of cluster: '+str(cluster[clus-2]),cmap=None,origin='upper')
        tools.save_fig('reorg_'+str(clus)+'_'+hemi,chemin+'/matrix')
