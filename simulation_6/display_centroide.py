# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 19/04/2021
@version: 2.0
@Recommandation: Python 3.7
@revision: 20/04/2021
@But: Centroide
@Step: 2
"""
import numpy as np
import matplotlib.pylab as plt

import display
import tools
import glob

#f_min=np.load('./data/L/495255_L_connectivity_withHKnob.npy')
#f_max=np.load('./data/L/121921_L_connectivity_withHKnob.npy')
#display.show_dot(f_min,title='495255_L')
#display.show_dot(f_max,title='121921_L')

L_name=np.load('./variables_R/L_name.npy')
L_val=np.load('./variables_R/L_val.npy')
L_trie=[L_name[i][8:] for i in np.argsort(L_val)]

fig=plt.figure(figsize=(100,100))
for i in range(0,np.shape(L_trie)[0]):
    chemin='./data/R/'+L_trie[i]
    a=fig.add_subplot(10,10,i+1)
    xs=np.load(chemin)
    a.title.set_text(str(L_trie[i][:8])+'-'+str(np.shape(xs)[0]))
    plt.xlim(0,100) 
    plt.ylim(0,100)
    plt.scatter(xs[:,0],xs[:,1],marker = '+', s = 50)
plt.show()

fig=plt.figure(figsize=(100,100))
for i in range(0,np.shape(L_trie)[0]):
    chemin='./data/R/'+L_trie[i]
    a=fig.add_subplot(10,10,i+1)
    xs=np.load(chemin)
    _,_,Img_Xs=tools.estimate_pseudo_density(xs)
    a.title.set_text(str(L_trie[i][:8]))
    plt.xlim(0,100) 
    plt.ylim(0,100)
    plt.imshow(Img_Xs)
plt.show()