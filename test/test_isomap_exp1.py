# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 27/05/2021
@version: 1.25
@Recommandation: Python 3.7
@revision 02/06/21
@But : Afficher les données en respectant la géodésie des données
"""
import numpy as np
import sys
sys.path.insert(1,'../libs')

import pandas as pd
from  sklearn.manifold import Isomap
import matplotlib.pylab as plt

import tools, display, barycenter, process

# Directory
hemi='L'
source='../data/'+hemi+'/'

columns=np.load('../variables/'+hemi+'/matrix_columns.npy',allow_pickle=True)
index=np.load('../variables/'+hemi+'/matrix_index.npy',allow_pickle=True)
if not(all(columns==index)):#controle intégrité matrice distance
    print('erreur')
    
d=np.load('../variables/'+hemi+'/matrix.npy')
data=pd.DataFrame((d>0.01)*np.sqrt(d),index=index,columns=columns)

iso=Isomap(n_neighbors=2,n_components=1,metric='precomputed')
X_transformed=iso.fit_transform(data)
print(X_transformed.shape)


plt.figure()
#plt.plot([i[0] for i in X_transformed],[i[1] for i in X_transformed],'+')
plt.plot([i[0] for i in X_transformed],[0 for i in X_transformed],'+')
plt.show()