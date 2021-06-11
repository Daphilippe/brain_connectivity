# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 21/04/2021
@version: 2.00
@Recommandation: Python 3.7
@revision: 11/06/2021
@But: Centroide
"""
import sys
sys.path.insert(1,'../../libs')
import centroide

hemi='L'
source1='../../data/'+hemi
source2='../../variables/'+hemi

#Attention adapter la fonction dans la librarie selon les répertoires par size
if False:# take a lot of time
    centroide.compute(source1,source2)
if True:
    centroide.display(data_source=source1,centroide_source=source2,directory=source2)
    centroide.matrix(source2)