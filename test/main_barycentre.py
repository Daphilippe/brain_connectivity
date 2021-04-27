# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 21/04/2021
@version: 1.50
@Recommandation: Python 3.7
@revision: 27/04/2021
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
destination='test/barycentre/'

# Changement des données
measures_locations = []
measures_weights = []
for np_name in glob.glob(str(source)+'*.np[yz]'):
    measures_locations.append(np.load(np_name))
    measures_weights.append(ot.unif(len(measures_locations[-1])))
Nmax=np.shape(measures_locations)[0]

# Initialisation du profil type
LX=[int(i[len(destination):-4]) for i in glob.glob(str(destination)+'*.np[yz]') ]
if len(LX)!=0:
    # Chargement du profil type
    itermax=np.max(LX)
    X=np.load(destination+str(itermax)+'.npy')
    b=ot.unif(np.shape(X)[0])
else:
    # Création du profil type
    itermax=0
    X_init = np.load('.'+str(np.load(variables+'centroide.npy')).replace('\\','/'))  # centroide
    b=ot.unif(np.shape(X_init)[0])
    X=None

# Calcul du barycentre sur l'ensemble des sujets et on sauvegarde chaque étape de la convergence
if False:
    for i in range(0,100):
        if X is None:
            X = ot.lp.free_support_barycenter(measures_locations, measures_weights,X_init,b,numItermax=1)
        else:
            X_init=X
            X = ot.lp.free_support_barycenter(measures_locations, measures_weights,X_init,b,numItermax=1)
            
        # Sauvegarde
        tools.save_value(X,str(i+itermax+1),directory=destination)
        display.show_dot(X,title='Barycenter')
        tools.save_fig(str(i+itermax+1),directory=destination)

# Calcule du barycentre avec une convergence acceptable sujet par sujet : de manière itérative        
if True:
    for i in range(1,Nmax-itermax):# On calcule 99 barycentres si on a 100 sujets
        if X is None:
            # On prend les 2 premiers sujets de la liste position 0 et 1
            X = ot.lp.free_support_barycenter(measures_locations[:2], measures_weights[:2],X_init,b,numItermax=1000)
        else:
            X_init=X
            L_loc=[X_init,measures_locations[i+itermax]]#sujet suivant pour calcul du barycentre position 2
            L_w=[b,measures_weights[i+itermax]]
            X = ot.lp.free_support_barycenter(L_loc, L_w,X_init,b,weights=np.array([(i+itermax)/(i+itermax+1),1/(i+itermax+1)]),numItermax=1000)  
        # Sauvegarde
        tools.save_value(X,str(i+itermax),directory=destination)
        display.show_dot(X,title='Barycenter')
        tools.save_fig(str(i+itermax),directory=destination)
sys.exit()