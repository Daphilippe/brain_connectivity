# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 08/04/2021
@version: 1.0
@Recommandation: Python 3.7
@But: Outil de controle
"""
import numpy as np
import ot

import tools
import display
import process
import sys

import matplotlib.pylab as plt

xs=np.load('try 3/xs.npy')
xt=np.load('try 3/xt.npy')

a, b = ot.unif(len(xs)), ot.unif(len(xt))
               
cols,colt=tools.label_position(xs,xt)
if False:
    display.plot_dots2(xs,xt,xs_color=cols,xt_color=colt,size=(20,20))
    
    #M=np.load('try 3/M.npy')
    #G0=np.load('try 3/G0.npy')
    #display.show_dot(M)
    #display.show_map(G0)
    
    M = ot.dist(xs,xt)
    G0 = ot.emd(a, b, M, numItermax=1000000)
    
    affixe='G0'
    directory='try 3/'
    pos1,w1=tools.degree(xs,xt,G0,degree=1)
    tools.save_value(pos1,'pos1_'+str(affixe),directory+str(affixe))
    tools.save_value(w1,'w1_'+str(affixe),directory+str(affixe))
    
    pos1_G0=np.load('try 3/G0/pos1_G0.npy')
    w1_G0=np.load('try 3/G0/w1_G0.npy')
    
    pos1_G0bis=pos1_G0.copy() #reverse sans recalcule du transport inverse
    pos1_G0bis[:, 0], pos1_G0bis[:, 1] = pos1_G0[:, 1], pos1_G0[:, 0]
    
    serie='Test'
    process.auto_affichage_discret(xs,xt,M,G0,pos1_G0,w1_G0,serie=serie)
    process.auto_affichage_continue(xs,a,xt,b,M,G0,pos=pos1_G0,weight=w1_G0,serie=serie,label='both')

racine='Test'
affixe='G0'
M = ot.dist(xs, xt)
G0 = ot.emd(a, b, M, numItermax=1000000)
process.auto_processing(xs,a,xt,b,G0,racine,affixe)  

pos1_G0=np.load(str(racine)+'/'+str(affixe)+'/pos1_'+str(affixe)+'.npy')
w1_G0=np.load(str(racine)+'/'+str(affixe)+'/w1_'+str(affixe)+'.npy')

process.auto_affichage_continue(xs,a,xt,b,M,G0,pos1_G0,w1_G0,str(racine)+'/'+str(affixe)+'/images',label='both')

affixe='G1'
process.auto_affichage_discret(xs,xt,M,G0,pos1_G0,w1_G0,str(racine)+'/'+str(affixe)+'/images')

sys.exit()