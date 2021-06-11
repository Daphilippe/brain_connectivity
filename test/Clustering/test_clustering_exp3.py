# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 26/05/21
@version: 2.00
@Recommandation: Python 3.7
@revision: 11/06/21
@But: Comparaison K-medoid et Wards linckage
Il faut regénérer les labels de Wards si on veut utiliser ce fichier 
expérience précédente : test_clustering_exp2.py
"""

import numpy as np

import sys
import pandas as pd
from sklearn import preprocessing

sys.path.insert(1,'../libs')
import tools, display

hemi='L'
source1='../../variables/'
chemin="../../variables/clustering/"+hemi+'/'

d=np.load(source1+hemi+'/matrix.npy')
index=np.load(source1+hemi+'/matrix_index.npy',allow_pickle=True)
columns=np.load(source1+hemi+'/matrix_columns.npy',allow_pickle=True)
data=pd.DataFrame((d>0.01)*np.sqrt(d),index=index,columns=columns)

# chargement des données générées
profils=list(range(len(columns)))

for i in range(0,2):
    cluster=i+2
    a=np.load(chemin+'labels.npy')[i]
    b=np.load(chemin+'labels_wards.npy')[i]
    
    lb = preprocessing.LabelBinarizer()
    a=lb.fit_transform(a)
    
    lb = preprocessing.LabelBinarizer()
    b=lb.fit_transform(b)

    #correlation
    corr=np.corrcoef(a,b,False)
    print(corr)

if True:#affichage
    i=1#1
    a=np.load(chemin+'labels.npy')[i]
    b=np.load(chemin+'labels_wards.npy')[i]
    
    df1=data.copy()
    columns = [df1.columns.tolist()[i] for i in list(np.argsort(a))]
    df1 = df1.reindex(columns, axis='columns')
    df1 = df1.reindex(columns, axis='index')
    
    df2=data.copy()
    columns = [df2.columns.tolist()[i] for i in list(np.argsort(b))[::-1]]#corr(a,b)
    df2 = df2.reindex(columns, axis='columns')
    df2 = df2.reindex(columns, axis='index')
    
    display.plot_map2(df1**2,df2**2,'K-medoids','Wards linckage',cmap=None,origin='upper')