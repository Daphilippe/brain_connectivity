# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 15/04/2021
@version: 1.0
@Recommandation: Python 3.7
@revision: 14/04/2021
@But: Centroide
"""
import numpy as np
import ot

import tools
import time
from operator import itemgetter

import glob
import sys

print('Begin')
numpy_vars = {}
for np_name in glob.glob('./data/R/*.np[yz]'):
    numpy_vars[np_name] = np.load(np_name)

Lval=[]
Lname=[]
for i in numpy_vars:
    L2_val=[]
    L2_name=[]
    #tic = time.clock()
    for j in numpy_vars:
        # xs = numpy_vars[i]
        # a = ot.emd(ot.unif(len(numpy_vars[i]))
        # xt = numpy_vars[j]
        # b = ot.emd(ot.unif(len(numpy_vars[j]))
        # M = ot.dist(xs,xt)
        # G = ot.emd(a,b,M, numItermax=1000000))
        # pos1_G,_= tools.degree(xs,xt,G,degree=1) # \Sigma(w_ij*d_ij) #implémenter ça (cf calcul matriciel)
        M=ot.dist(numpy_vars[i],numpy_vars[j])
        G=ot.emd(ot.unif(len(numpy_vars[i])), ot.unif(len(numpy_vars[j])),M,numItermax=1000000)
        #tools.save_value(G,'G_emd_'+str(j[9:17]),directory='variables/'+str(i[9:17]))#attention à la taille
        cost=G*M    
        L2_name.append(str(j[9:17]))
        L2_val.append(np.sum(cost))# mettre au carrée
    tools.save_value(L2_val,'L2_val_'+str(i[9:17]),directory='variables/'+str(i[9:17]))
    tools.save_value(L2_name,'L2_name_'+str(i[9:17]),directory='variables/'+str(i[9:17]))
    Lval.append(np.sum(L2_val))
    Lname.append(i)
    #toc = time.clock()
    #print('\n Time: ',toc-tic)
argmin=np.argmin(Lval)
centroide=Lname[argmin]
print(centroide)
tools.save_value(Lval,'L_val',directory='variables')
tools.save_value(Lname,'L_name',directory='variables')
tools.save_value(centroide,'centroide',directory='variables')

L=list(zip(Lname,Lval))
L=np.array(L)
L[:,0]=[i[2:] for i in L[:,0]]
L=np.array(sorted(L,key=itemgetter(1)))
print('End')
sys.exit()