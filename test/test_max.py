# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 11/05/2021
@version: 1.00
@Recommandation: Python 3.7
"""
import numpy as np
import sys
import cv2

sys.path.insert(1,'../libs')
import tools,display
from skimage.morphology import local_maxima

import matplotlib.pylab as plt

    
# Chemins
source='../data/L/'
variables='../variables/L/'
destination='max'
size=len(source)-1

X=np.load('test/L/mean.npy')

mask1=np.array([[0,1,0],
               [1,1,1],
               [0,1,0]])


mask=np.array([[1,1,1],
                [1,1,1],
                [1,1,1]])

r=local_maxima(X,selem=mask)
display.plot_map2(X,r,sub1title='Moyenne',sub2title='Maximum')


X=np.load('test/L/barycentre.npy')
_,_,data=tools.estimate_pseudo_density(X)
data=data/np.max(data)

r=local_maxima(data,selem=mask)
display.plot_map2(data,r,sub1title='Barycentre',sub2title='Maximum')
