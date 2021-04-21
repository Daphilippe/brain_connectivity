# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@revision: 21/04/2021
@version: 1.75
@Recommandation: Python 3.7
@revision: 21/04/2021
@But: Centroide
"""
import sys
sys.path.insert(1,'../libs')
import centroide

if False:# take a lot of time
    centroide.compute(source='../data/L/',directory='./variables/')
if True:
    centroide.display(data_source='../data/L',centroide_source='../variables/L',directory='../variables/L')
    centroide.display(data_source='../data/R',centroide_source='../variables/R',directory='../variables/R')
    centroide.matrix('../variables/L/')
    centroide.matrix('../variables/R/')