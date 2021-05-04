# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 21/04/2021
@version: 1.75
@Recommandation: Python 3.7
@revision: 28/04/2021
@But: barycentre
"""
import numpy as np
import sys
import glob
import ot

sys.path.insert(1,'../libs')
import tools, display, barycenter

# Chemins
source='../data/R/'
variables='../variables/R/'
destination='barycentre_R/'

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
    X_init=None
else:
    # Création du profil type
    itermax=0
    #X_init = np.load('.'+str(np.load(variables+'centroide.npy')).replace('\\','/'))  # centroide
    #b=ot.unif(np.shape(X_init)[0])
    k = int(np.mean([np.shape(i)[0] for i in measures_locations]))# number of Diracs of the barycenter
    X_init = np.random.normal(0., 1., (k, 2))  # initial Dirac locations
    b = np.ones((k,)) / k  # weights of the barycenter (it will not be optimized, only the locations are optimized)
    X=None

if False:# Prend du temps     
    #Calcul du barycentre itératif
    barycenter.iterative_barycenter(X,X_init,b,measures_locations,measures_weights,Nmax,itermax,stopThr=0.01,destination=destination)    

# Sauvegarde des images
for np_name in glob.glob(str(destination)+'*.np[yz]'):
    display.show_dot(np.load(np_name),title='Barycenter')
    tools.save_fig('dot_'+np_name.replace('\\','/')[len(destination):-4],directory=destination+'Dot')

    _,_,Img_xs=tools.estimate_pseudo_density(np.load(np_name))
    display.show_map(Img_xs,title='Barycenter')
    tools.save_fig('map_'+np_name.replace('\\','/')[len(destination):-4],directory=destination+'Map')

sys.exit()