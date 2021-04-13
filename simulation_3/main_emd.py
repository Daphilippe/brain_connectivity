# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 08/04/2021
@version: 1.0
@Recommandation: Python 3.7
@But: Régularisation
"""
# -*- coding: utf-8 -*-
import numpy as np
import ot

import tools
import display
import process
import sys
    
racine='continue_1'
# Création des données
source_dot=[]
source_map=[]
for i in range(2):
    # Génération des données
    temp=np.random.randint(2,5)
    S=tools.data_generator_simulation2(rand_seed=temp,noise=0.5*np.log(temp))
    xs,a=S
    source_dot.append([xs,a])
    
    # Génération de la carte
    _,_,Img_Xs=tools.estimate_pseudo_density(xs)
    source_map.append(Img_Xs)

for i in range(len(source_map)):
    display.show_map(source_map[i])
    tools.save_fig('source_'+str(i),directory=racine)

def f(G):
    #Frobeniusregularization
    return 0.5 * np.sum(G**2)

def df(G):
    return G

if True:
    ## Test régularisation
    # Processing
    Xs,a=source_dot[0]
    tools.save_value(Xs,'xs',directory=racine)
    tools.save_value(a,'a',directory=racine)
    Img_Xs=source_map[0]
    Img2Xs,aa=tools.extract_point(Img_Xs)
    
    Xt,b=source_dot[1]
    tools.save_value(Xt,'xt',directory=racine)
    tools.save_value(b,'b',directory=racine)
    Img_Xt=source_map[1]
    Img2Xt,bb=tools.extract_point(Img_Xt)
    
    # Cost matrix
    M = ot.dist(Img2Xs, Img2Xt)
    tools.save_value(M,'M',directory=racine)
    
    # Double 
    print('Regularisation entropique')
    reg_lasso=10
    reg_entropique=10
    temp2=ot.optim.gcg(aa, bb, M, reg_entropique, reg_lasso, f, df)
    process.auto_processing(Img2Xs,aa,Img2Xt,bb,temp2,str(racine),'2reg_'+str(reg_entropique)+'_'+str(reg_lasso))

    pos1_G0=np.load('try 3/G0/pos1_G0.npy')
    w1_G0=np.load('try 3/G0/w1_G0.npy')
    
    process.auto_affichage_continue(Img2Xs,aa,Img2Xt,bb,M,temp2,serie=str(racine)+'/2reg_'+str(reg_entropique)+'_'+str(reg_lasso)+'/images',label='both')
    
    # Sans régularisation
    print('Sans regularisation')
    G0 = ot.emd(aa, bb, M, numItermax=1000000)
    process.auto_processing(Img2Xs,aa,Img2Xt,bb,G0,racine,'G0')  
    
    pos1_G0=np.load('try 3/G0/pos1_G0.npy')
    w1_G0=np.load('try 3/G0/w1_G0.npy')
    
    process.auto_affichage_continue(Img2Xs,aa,Img2Xt,bb,M,G0,str(racine)+'/G0/images',label='both')
        
sys.exit()