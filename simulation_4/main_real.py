# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 14/04/2021
@version: 1.0
@Recommandation: Python 3.7
@revision: 14/04/2021
@But: Données réelles / mesocentre
"""
import numpy as np
import ot

import tools
import process
import sys

racine_data='data/L'    
sujet1='100206_L_connectivity_withHKnob.npy'
sujet2='103111_L_connectivity_withHKnob.npy'

# Discret
xs=np.load(racine_data+sujet1)
a=ot.unif(len(xs))
xt=np.load(racine_data+sujet2)
b=ot.unif(len(xt))
# Continue
_,_,Img_Xs=tools.estimate_pseudo_density(xs)
Img2Xs,aa=tools.extract_point(Img_Xs)
_,_,Img_Xt=tools.estimate_pseudo_density(xt)
Img2Xt,bb=tools.extract_point(Img_Xt)

racine='test'
Md = ot.dist(xs,xt)
Mc = ot.dist(Img2Xs, Img2Xt)
if False:
    # Discret
    G = ot.emd(a, b, Md, numItermax=1000000)
    process.automatisation_discret(xs,a,xt,b,Md,G,racine,'G')
    
    # Continue    
    Gc = ot.emd(aa, bb, Mc, numItermax=1000000)
    process.automatisation_continue(Img2Xs,aa,Img2Xt,bb,Mc,Gc,racine,'Gc')

if False:
    print('Regularisation entropique')
    def f(G):
        #Frobeniusregularization
        return 0.5 * np.sum(G**2)
    
    def df(G):
        return G
    
    reg_lasso=10
    reg_entropique=10
    temp2=ot.optim.gcg(aa, bb, Mc, reg_entropique, reg_lasso, f, df)
        
    affixe= '2reg_'+str(reg_entropique)+'_'+str(reg_lasso)
    process.automatisation_continue(Img2Xs,aa,Img2Xt,bb,Mc,temp2,racine,affixe)
sys.exit()