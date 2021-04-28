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
destination='barycentre/'

size=len(source)-1
# Changement des données
measures_locations = []
measures_weights = []


L_name=np.load(str(variables)+'L_name.npy')
L_val=np.load(str(variables)+'L_val.npy')
L_trie=[L_name[i][size:] for i in np.argsort(L_val)]
for np_name in  L_trie:#glob.glob(str(source)+'*.np[yz]'):
    np_name=source+np_name
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
if False:
    for i in range(1,Nmax-itermax):# On calcule 99 barycentres si on a 100 sujets
        if X is None:
            # On prend les 2 premiers sujets de la liste position 0 et 1
            X = ot.lp.free_support_barycenter(measures_locations[:2], measures_weights[:2],X_init,b)
        else:
            X_init=X
            L_loc=[X_init,measures_locations[i+itermax]]#sujet suivant pour calcul du barycentre position 2
            L_w=[b,measures_weights[i+itermax]]
            X = ot.lp.free_support_barycenter(L_loc, L_w,X_init,b,weights=np.array([(i+itermax)/(i+itermax+1),1/(i+itermax+1)]),numItermax=100)  
        # Sauvegarde
        tools.save_value(X,str(i+itermax),directory=destination)
        display.show_dot(X,title='Barycenter')
        tools.save_fig(str(i+itermax),directory=destination)
        
# Solution paufinée        
def free_support_barycenter(measures_locations, measures_weights, X_init, b=None, weights=None, numItermax=100,
                            stopThr=1e-7):
    iter_count = 0
    N = len(measures_locations)
    k = X_init.shape[0]
    d = X_init.shape[1]
    
    if b is None:
        b = np.ones((k,)) / k
    if weights is None:
        weights = np.ones((N,)) / N

    X = X_init
    displacement_norm = stopThr + 1.

    while (displacement_norm > stopThr and iter_count < numItermax):

        T_sum = np.zeros((k, d))

        for (measure_locations_i, measure_weights_i, weight_i) in zip(measures_locations, measures_weights,weights.tolist()):
            M_i = ot.dist(X, measure_locations_i)
            T_i = ot.emd(b, measure_weights_i, M_i,numItermax=1000000)
            T_sum = T_sum + weight_i * np.reshape(1. / b, (-1, 1)) * np.matmul(T_i, measure_locations_i)
        displacement_norm = np.sqrt((np.sum(np.square(T_sum - X))))
        print(displacement_norm)
        X = T_sum
        iter_count += 1
    return X

if True:
    for i in range(1,Nmax-itermax):# On calcule 99 barycentres si on a 100 sujets
        if X is None:
            # On prend les 2 premiers sujets de la liste position 0 et 1
            X = free_support_barycenter(measures_locations[:2], measures_weights[:2],X_init,b,stopThr=1e-2)
        else:
            X_init=X
            L_loc=[X_init,measures_locations[i+itermax]]#sujet suivant pour calcul du barycentre position 2
            L_w=[b,measures_weights[i+itermax]]
            X = free_support_barycenter(L_loc, L_w,X_init,b,weights=np.array([(i+itermax)/(i+itermax+1),1/(i+itermax+1)]),stopThr=1e-2)  
        # Sauvegarde
        tools.save_value(X,str(i+itermax),directory=destination)
        display.show_dot(X,title='Barycenter')
        tools.save_fig(str(i+itermax),directory=destination)
sys.exit()