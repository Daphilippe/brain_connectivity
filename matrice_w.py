# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 19/04/2021
@version: 1.75
@Recommandation: Python 3.7
@revision: 21/04/2021
@But: clustering
@Step: 1
"""
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

import glob
import sys
sys.path.insert(1,'../libs')
import tools

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

sys.exit()