# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 19:51:15 2021

@author: Daphilippe
"""
import numpy as np
import matplotlib.pylab as plt

import display
import tools
import glob

f_min=np.load('./data/L/495255_L_connectivity_withHKnob.npy')
f_max=np.load('./data/L/121921_L_connectivity_withHKnob.npy')
#display.show_dot(f_min,title='495255_L')
#display.show_dot(f_max,title='121921_L')

L_trie=np.load('./variables/L_trie.npy')

fig=plt.figure(figsize=(100,100))
for i in range(0,np.shape(L_trie)[0]):
    chemin='./data/L/'+L_trie[i][0]
    a=fig.add_subplot(10,10,i+1)
    xs=np.load(chemin)
    a.title.set_text(str(L_trie[i][0][:8])+'-'+str(np.shape(xs)[0]))
    plt.xlim(0,100) 
    plt.ylim(0,100)
    plt.scatter(xs[:,0],xs[:,1],marker = '+', s = 50)
plt.show()


fig=plt.figure(figsize=(100,100))
for i in range(0,np.shape(L_trie)[0]):
    chemin='./data/L/'+L_trie[i][0]
    a=fig.add_subplot(10,10,i+1)
    xs=np.load(chemin)
    _,_,Img_Xs=tools.estimate_pseudo_density(xs)
    a.title.set_text(str(L_trie[i][0][:8]))
    plt.xlim(0,100) 
    plt.ylim(0,100)
    plt.imshow(Img_Xs)
plt.show()