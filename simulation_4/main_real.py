# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 08/04/2021
@version: 1.0
@Recommandation: Python 3.7
@But: Données réelles
"""
import numpy as np
import ot

import tools
import process
import sys

racine_data='data/'    
sujet='100206_L_connectivity_withHKnob.npy'
xs=np.load(racine_data+sujet)
a=ot.unif(len(xs))
sujet='103111_L_connectivity_withHKnob.npy'
xt=np.load(racine_data+sujet)
b=ot.unif(len(xt))

# Discret
M = ot.dist(xs,xt)
G = ot.emd(a, b, M, numItermax=1000000)

process.automatisation_discret(xs,a,xt,b,M,G,'test','G')

# Continue
_,_,Img_Xs=tools.estimate_pseudo_density(xs)
Img2Xs,aa=tools.extract_point(Img_Xs)
_,_,Img_Xt=tools.estimate_pseudo_density(xt)
Img2Xt,bb=tools.extract_point(Img_Xt)

Mc = ot.dist(Img2Xs, Img2Xt)
Gc = ot.emd(aa, bb, Mc, numItermax=1000000)

process.automatisation_continue(Img2Xs,aa,Img2Xt,bb,Mc,Gc,'test','Gc')
sys.exit()