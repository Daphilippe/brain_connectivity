# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 26/05/21
@version: 1.00
@Recommandation: Python 3.7
@But: Comparaison K-medoid et Wards linckage
"""

import numpy as np

import sys
import pandas as pd
import matplotlib.pylab as plt
from sklearn import preprocessing

sys.path.insert(1,'../libs')
import tools as tools, display

chemin="../variables/clustering/"
hemi='L'
d=np.load('../variables/'+hemi+'/matrix.npy')
index=np.load('../variables/'+hemi+'/matrix_index.npy',allow_pickle=True)
columns=np.load('../variables/'+hemi+'/matrix_columns.npy',allow_pickle=True)
data=pd.DataFrame((d>0.01)*np.sqrt(d),index=index,columns=columns)

# chargement des données générées
profils=list(range(len(columns)))

for i in range(2):
    cluster=i+2
    a=np.load(chemin+'labels.npy')[i]
    b=np.load(chemin+'labels_wards.npy')[i]
    
    if cluster!=2:
        lb = preprocessing.LabelBinarizer()
        a=lb.fit_transform(a)
        
        lb = preprocessing.LabelBinarizer()
        b=lb.fit_transform(b)

    #correlation
    corr=np.corrcoef(a,b)

if False:#affichage
    df1=data.copy()
    columns = [df1.columns.tolist()[i] for i in list(np.argsort(a))]
    df1 = df1.reindex(columns, axis='columns')
    df1 = df1.reindex(columns, axis='index')
    
    df2=data.copy()
    columns = [df2.columns.tolist()[i] for i in list(np.argsort(b))]
    df2 = df2.reindex(columns, axis='columns')
    df2 = df2.reindex(columns, axis='index')
    
    display.plot_map2(df1,df2,'K-medoids','Wards linckage')