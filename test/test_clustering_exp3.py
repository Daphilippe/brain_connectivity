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

sys.path.insert(1,'../libs')
import tools as tools

chemin="../variables/clustering/"
hemi='L'
d=np.load('../variables/'+hemi+'/matrix.npy')
index=np.load('../variables/'+hemi+'/matrix_index.npy',allow_pickle=True)
columns=np.load('../variables/'+hemi+'/matrix_columns.npy',allow_pickle=True)


# chargement des données générées
profils=list(range(len(columns)))
a=np.load(chemin+'label_wards_clusters_2.npy')-1
b=np.load(chemin+'labels.npy')[0]

#count labels
print('a :',np.sum(a),100-np.sum(a))
print('b :',np.sum(b),100-np.sum(b))

#correlation
print(np.corrcoef(a,b))
print(np.corrcoef(a,1-b))