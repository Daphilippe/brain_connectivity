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
from scipy.spatial.distance import sqeuclidean

import glob
import sys

print('Begin')
numpy_vars = {}
for np_name in glob.glob('./data/L/*.np[yz]'):
    numpy_vars[np_name] = np.load(np_name)

Lval=[]
Lname=[]
for i in numpy_vars:
    L2=[]
    for j in numpy_vars:
        if i==j:
            continue
        else:
            # xs = numpy_vars[i]
            # a = ot.emd(ot.unif(len(numpy_vars[i]))
            # xt = numpy_vars[j]
            # b = ot.emd(ot.unif(len(numpy_vars[j]))
            # M = ot.dist(xs,xt)
            # G = ot.emd(a,b,M, numItermax=1000000))
            # pos1_G,_= tools.degree(xs,xt,G,degree=1) # \Sigma(w_ij*d_ij) #implémenter ça (cf calcul matriciel)
            pos1_G,_=tools.degree(numpy_vars[i],numpy_vars[j],ot.emd(ot.unif(len(numpy_vars[i])), ot.unif(len(numpy_vars[j])), ot.dist(numpy_vars[i],numpy_vars[j]), numItermax=1000000),degree=1)
            # On déterminer la distance entre les points utilisés dans la correspondance entre la source i et la cible j
            # On additionne l'ensemble des distances nécessaire
            L2.append(np.sum(  [sqeuclidean(numpy_vars[i][pos1_G[x,1]],numpy_vars[j][pos1_G[x,0]]) for x in range(np.shape(pos1_G)[0])] ))
    Lval.append(np.sum(L2))
    Lname.append(i)
argmin=np.argmin(Lval)
barycentre=Lname[argmin]
print(barycentre)

tools.save_value(Lval,'L_val',directory='variables')
tools.save_value(Lname,'L_name',directory='variables')
tools.save_value(barycentre,'barycentre',directory='variables')

print('End')
sys.exit()