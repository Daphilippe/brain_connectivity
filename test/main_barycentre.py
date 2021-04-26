# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 21/04/2021
@version: 1.25
@Recommandation: Python 3.7
@revision: 22/04/2021
@But: barycentre
"""
import numpy as np
import sys
import glob
import ot
import matplotlib.pylab as plt
import time

sys.path.insert(1,'../libs')
import tools, display, process


source='../data/L/'
variables='../variables/L/'

measures_locations = []
measures_weights = []
N=1
Nmax=100
for np_name in glob.glob(str(source)+'*.np[yz]'):
    measures_locations.append(np.load(np_name))
    measures_weights.append(ot.unif(len(measures_locations[-1])))
    if N>=Nmax:
        break
    N=N+1

nb_dot=int(np.mean([np.shape(i)[0] for i in measures_weights]))
X_init = np.load('.'+str(np.load(variables+'centroide.npy')).replace('\\','/'))  # centroide
b=ot.unif(np.shape(X_init)[0])
#b= np.ones((nb_dot,))/nb_dot # weights of the barycenter (it will not be optimized, only the locations are optimized)

Ltime=[]
i=Nmax
#for i in range(2,Nmax):
t1=time.time()
X = ot.lp.free_support_barycenter(measures_locations[:i], measures_weights[:i],X_init,b,numItermax=1)
tools.save_value(X,str(i),directory='test')


if False:
    plt.figure()
    for (x_i, b_i) in zip(measures_locations[:i], measures_weights[:i]):
        color = np.random.randint(low=1, high=10 * N)
        plt.scatter(x_i[:, 0], x_i[:, 1], s=b_i * 1000, label='input measure')
    plt.scatter(X[:, 0], X[:, 1], s=b * 1000, c='black', marker='^', label='2-Wasserstein barycenter')
    plt.title('Data measures and their barycenter')
    plt.legend(loc=0)
    tools.save_fig(str(i),directory='test')
t2=time.time()

Ltime.append(t2-t1)
print(i,Ltime[-1])
    
sys.exit()