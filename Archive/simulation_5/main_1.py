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
import time


def f(G):
    #Frobeniusregularization
    return 0.5 * np.sum(G**2)

def df(G):
    return G

print('Begin')
racine_data='data/L/'  

sujet1='100206_L_connectivity_withHKnob.npy'
sujet2='103111_L_connectivity_withHKnob.npy'

xs=np.load(racine_data+sujet1)
a=ot.unif(len(xs))
xt=np.load(racine_data+sujet2)
b=ot.unif(len(xt))

Md = ot.dist(xs,xt)

tic = time.clock()
racine='G_emd'
affixe='G'
temp = ot.emd(a, b, Md, numItermax=1000000)
process.automatisation_discret(xs,a,xt,b,Md,temp,racine,affixe)
toc = time.clock()
print('\nSans regularisation',toc - tic)


tic = time.clock()
racine='G_sink'
reg_entropique=0.001
affixe= '1reg_'+str(reg_entropique)

temp=ot.sinkhorn(a, b, Md, reg_entropique)
process.automatisation_discret(xs,a,xt,b,Md,temp,racine,affixe)
toc = time.clock()
print('\nAvec reg1=0.001',toc - tic)

tic = time.clock()
racine='G_sink'
reg_entropique=0.01
affixe= '1reg_'+str(reg_entropique)

temp=ot.sinkhorn(a, b, Md, reg_entropique)
process.automatisation_discret(xs,a,xt,b,Md,temp,racine,affixe)
toc = time.clock()
print('\nAvec reg1=0.01',toc - tic)

tic = time.clock()
racine='G_sink_frob'
reg_lasso=0.1
reg_entropique=0.1
affixe= '2reg_'+str(reg_entropique)+'_'+str(reg_lasso)

temp=ot.optim.gcg(a, b, Md, reg_entropique, reg_lasso, f, df)
process.automatisation_discret(xs,a,xt,b,Md,temp,racine,affixe)
toc = time.clock()
print('\nAvec reg1=0.01, reg2=1',toc - tic)

print('End')
sys.exit()