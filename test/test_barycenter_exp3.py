# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 14/05/2021
@version: 1.00
@Recommandation: Python 3.7
@But: comparaison barycentre avancées
"""
import numpy as np
import sys
import glob
import ot
import random


from time import process_time

sys.path.insert(1,'../libs')
import tools, display