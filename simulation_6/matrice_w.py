# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 19/04/2021
@version: 2.0
@Recommandation: Python 3.7
@revision: 20/04/2021
@But: Centroide
@Step: 3
"""
import numpy as np

import tools

import glob
import sys
import pandas as pd
import matplotlib.pylab as plt

### Right
numpy_vars = {}
for i in list(zip(glob.glob('./variables_R/*/L2_name*.np[yz]'),glob.glob('./variables_R/*/L2_val*.np[yz]'))):
    name,val=i
    numpy_vars[name[14:22]] = np.load(val)
    df = pd.DataFrame(data=numpy_vars,index=np.load(name))
    
df.sort_index()
df = df.reindex(sorted(df.columns), axis=0)

tools.save_value(df,'matrix_R','./variables_R/')
tools.save_value(df.index,'matrix_R_index','./variables_R/')
tools.save_value(df.columns,'matrix_R_columns','./variables_R/')
##df.to_csv(path_or_buf='variables_R/matrix_R.csv')

plt.imshow(df)
plt.show()

### Left
numpy_vars = {}
for i in list(zip(glob.glob('./variables_L/*/L2_name*.np[yz]'),glob.glob('./variables_L/*/L2_val*.np[yz]'))):
    name,val=i
    numpy_vars[name[14:22]] = np.load(val)
    df = pd.DataFrame(data=numpy_vars,index=np.load(name))
    
df.sort_index()
df = df.reindex(sorted(df.columns), axis=0)

tools.save_value(df,'matrix_L','./variables_L/')
tools.save_value(df.index,'matrix_L_index','./variables_L/')
tools.save_value(df.columns,'matrix_L_columns','./variables_L/')
##df.to_csv(path_or_buf='variables_R/matrix_R.csv')
plt.imshow(df)
plt.show()

sys.exit()