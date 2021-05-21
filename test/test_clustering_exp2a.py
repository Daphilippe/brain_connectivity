# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 20/05/21
@version: 1.00
@Recommandation: Python 3.7
@But: determiner le nb de cluster (clustrering hiearchique)
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

if True:# clustering hierarchique
    L=[]
    L_label=[]
    import scipy.cluster.hierarchy as sch
    df=data.copy()
    dist=sch.ward(np.sqrt(df))
    
    seuil=0.95
    for cluster in range(1,101):
        #label=sch.fcluster(dist,cluster,criterion='distance')
        label=sch.fcluster(dist,cluster,criterion='maxclust')
        arg=np.argsort(label)
        if len(L)==0:#initialisation
            L.append((arg,cluster))#ajout de la réorganisation des colonnes
            L_label.append( (label,cluster))
        depassement=0
        for i in L:
            similarity=np.sum( ((i[0]-arg)==0)*1 )/(np.shape(df)[0])
            if  similarity < seuil:# similitude de moins de  seuil% des précédents
                depassement=depassement+1#comptage des dépassements tolérances
        if depassement==len(L):# actualisation de la liste
            L.append((arg,cluster))
            L_label.append( (label,cluster))


for i in L:
    df=data.copy()
    columns = [df.columns.tolist()[i] for i in list(i[0])]
    df = df.reindex(columns, axis='columns')
    df = df.reindex(columns, axis='index')
    
    plt.figure()
    plt.imshow(df)
    plt.title('Nombre de cluster : '+str(i[1]))
    plt.show()