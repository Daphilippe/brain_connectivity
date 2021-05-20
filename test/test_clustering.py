# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 20/05/21
@version: 1.00
@Recommandation: Python 3.7
@But: determiner le nb de cluster
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
    import scipy.cluster.hierarchy as sch
    df=data.copy()
    dist=sch.ward(np.sqrt(df))
    
    for cluster in range(2,101):
        label=sch.fcluster(dist,cluster,criterion='distance')
        if len(L)==0:
            L.append((label,cluster))
        if np.sum( ( (L[-1][0]-label)!=0)*1 ) > 5:# similitude à 95% du précédent
            L.append((label,cluster))