# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 21/04/2021
@version: 1.00
@Recommandation: Python 3.7
@revision: 22/04/2021
@But: test
"""
import numpy as np
import sys
import ot

sys.path.insert(1,'../libs')
import tools

(xs,a),_=tools.data_generator_simulation1()
tools.save_value(xs,'xs',directory='./temp')
tools.save_value(a,'a',directory='./temp')

distance=[]
for i in range(0,100):
    xt=np.array([ j+[i,0] for j in np.load('./temp/xs.npy')])
    b=np.load('./temp/a.npy')
    M=ot.dist(xs,xt)
    G=ot.emd(a, b, M, numItermax=1000000)
    distance.append(np.sqrt(np.sum(G*M)))
    
    #cols,_=tools.label_position(xs,xt)
    #pos1,w1=tools.degree(xs,xt,G,degree=1)
    #display.plot_dots_links(xs,xt,pos1,w1,xs_color=cols)
    #tools.save_fig('links',directory='./temp/'+str(i))

sys.exit()