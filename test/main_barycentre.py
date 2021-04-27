# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 21/04/2021
@version: 1.25
@Recommandation: Python 3.7
@revision: 22/04/2021
@But: barycentre
"""
import numpy as np
import sys
import glob
import ot

sys.path.insert(1,'../libs')
import tools, display

# Chemins
source='../data/L/'
variables='../variables/L/'
destination='test/barycentre 100/'

# Changement des données
measures_locations = []
measures_weights = []
N=1
Nmax=100
for np_name in glob.glob(str(source)+'*.np[yz]'):
    measures_locations.append(np.load(np_name))
    measures_weights.append(ot.unif(len(measures_locations[-1])))
    if N>=Nmax:
        break
    N=N+1

# Initialisation du profil type
LX=[int(i[len(destination):-4]) for i in glob.glob(str(destination)+'*.np[yz]') ]
if len(LX)!=0:
    # Chargement du profil type
    itermax=np.max(LX)
    X=np.load(destination+str(itermax)+'.npy')
    b=ot.unif(np.shape(X)[0])
else:
    # Création du profil type
    X_init = np.load('.'+str(np.load(variables+'centroide.npy')).replace('\\','/'))  # centroide
    b=ot.unif(np.shape(X_init)[0])
    X=None

# Calcul du barycentre    
for i in range(0,100):
    if X is None:
        X = ot.lp.free_support_barycenter(measures_locations, measures_weights,X_init,b,numItermax=1)
    else:
        X_init=X
        X = ot.lp.free_support_barycenter(measures_locations, measures_weights,X_init,b,numItermax=1)
        
    # Sauvegarde
    tools.save_value(X,str(i+itermax),directory=destination+str(2))
    display.show_dot(X,title='Barycenter')
    tools.save_fig(str(i+itermax+1),directory=destination+str(2))
sys.exit()