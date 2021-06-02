# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 27/05/2021
@version: 1.25
@Recommandation: Python 3.7
@revision 02/06/21
@But: KMedoids : calculer le barycentre associée à chaque k clusters
"""

import numpy as np
import sys
sys.path.insert(1,'../libs')

import tools, display, barycenter, process

# Directory
hemi='L'
source='../data/'+hemi+'/'
source2="../variables/clustering/"+hemi+"/"

columns=np.load('../variables/'+hemi+'/matrix_columns.npy',allow_pickle=True)
index=np.load('../variables/'+hemi+'/matrix_index.npy',allow_pickle=True)

if not(all(columns==index)):#controle intégrité matrice distance
    print('erreur')

# Load    
cluster=np.load(source2+'cluster.npy')
labels=np.load(source2+'labels.npy')
k=labels[1] #labels[1]

# Initialisation variables
c0=[]
w0=[]
c1=[]
w1=[]
c2=[]
w2=[]
nb_dot=500# mettre idéalement à 2000, ici pour accélérer les calculs aucunes incidences sur les résultats
for i in range(len(k)):
    if k[i]==0:
        c0.append(np.load(source+index[i]+'_connectivity_withHKnob.npy'))
        w0.append(  np.ones(np.shape(c0[-1])[0])/np.shape(c0[-1])[0] )
    if k[i]==1:
        c1.append(np.load(source+index[i]+'_connectivity_withHKnob.npy'))
        w1.append(  np.ones(np.shape(c1[-1])[0])/np.shape(c1[-1])[0] )
    if k[i]==2:
        c2.append(np.load(source+index[i]+'_connectivity_withHKnob.npy'))
        w2.append(  np.ones(np.shape(c2[-1])[0])/np.shape(c2[-1])[0] )
        
X_init = np.random.normal(0., 1., (nb_dot, 2))#2087 pour R nombre de point moyen de l'ensemble des profils     
b=np.ones(np.shape(X_init)[0])/np.shape(X_init)[0]

k='k3'
# Calcul et sauvegarde donnée
if False:
    X0=barycenter.iterative_barycenter(None,X_init,b,c0,w0,save=False,Nmax=len(w0))
    tools.save_value(X0, 'X0',source2+'barycentre_'+k)
    X1=barycenter.iterative_barycenter(None,X_init,b,c1,w1,save=False,Nmax=len(w1))
    tools.save_value(X1, 'X1',source2+'barycentre_'+k)
    X2=barycenter.iterative_barycenter(None,X_init,b,c2,w2,save=False,Nmax=len(w2))
    tools.save_value(X2, 'X2',source2+'barycentre_'+k)
    
if False:#barycentre des sous groupes
    X0=np.load(source2+'barycentre_'+k+'/X0.npy')
    _,_,Img_xs=tools.estimate_pseudo_density(X0)
    display.show_map(Img_xs,title='Barycenter cluster 0 - '+hemi)
    tools.save_fig('Barycenter cluster 0 - '+hemi, source2+'barycentre_'+k+'/')
    
    X1=np.load(source2+'barycentre_'+k+'/X1.npy')
    _,_,Img_xs=tools.estimate_pseudo_density(X1)
    display.show_map(Img_xs,title='Barycenter cluster 1 - '+hemi)
    tools.save_fig('Barycenter cluster 1 - '+hemi, source2+'barycentre_'+k+'/')
        
    X2=np.load(source2+'barycentre_'+k+'/X2.npy')
    _,_,Img_xs=tools.estimate_pseudo_density(X2)
    display.show_map(Img_xs,title='Barycenter cluster 2 - '+hemi)
    tools.save_fig('Barycenter cluster 2 - '+hemi, source2+'barycentre_'+k+'/')
    
if True:#maximum locaux
    destination = source2+'barycentre_'+k+'/'
    min_distance=1
    percent=90#seuillage des valeurs
    grid_size=101
    
    #cluster 0
    X='0'
    titre='Cluster: '+X+' - '+hemi
    save_title='max_'+hemi+'_'+X
    X0=np.load(source2+'barycentre_'+k+'/X0.npy')
    
    process.auto_max(X0,titre,save_title,destination,min_distance=1,percent=90,grid_size=101)
    
    # cluster 1
    X='1'
    titre='Cluster: '+X+' - '+hemi
    save_title='max_'+hemi+'_'+X
    X1=np.load(source2+'barycentre_'+k+'/X1.npy')
    process.auto_max(X1,titre,save_title,destination,min_distance=1,percent=90,grid_size=101)
        
    #cluster 2
    X='2'
    titre='Cluster: '+X+' - '+hemi
    save_title='max_'+hemi+'_'+X
    X2=np.load(source2+'barycentre_'+k+'/X2.npy')
    _,_,Img_xs=tools.estimate_pseudo_density(X2)
    process.auto_max(X2,titre,save_title,destination,min_distance=1,percent=90,grid_size=101)
    
