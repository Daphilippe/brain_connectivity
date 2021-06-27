# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 21/04/2021
@version: 1.00
@Recommandation: Python 3.7
@revision: 11/05/2021
@But: test
"""
import numpy as np
import sys
import ot

sys.path.insert(1,'../libs')
import tools,display,barycenter,process

"""
# Translation d'un nuage de point de manière uniforme et comparaison avec distance de Wasserstein
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
"""

"""
# Calcul du barycentre sur l'ensemble des sujets et on sauvegarde chaque étape de la convergence
if False:
    for i in range(0,100):
        if X is None:
            X = ot.lp.free_support_barycenter(measures_locations, measures_weights,X_init,b,numItermax=1)
        else:
            X_init=X
            X = ot.lp.free_support_barycenter(measures_locations, measures_weights,X_init,b,numItermax=1)
            
        # Sauvegarde
        tools.save_value(X,str(i+itermax+1),directory=destination)
        display.show_dot(X,title='Barycenter')
        tools.save_fig(str(i+itermax+1),directory=destination)
        
# Calcule du barycentre avec une convergence acceptable sujet par sujet : de manière itérative        
if False:
    for i in range(1,Nmax-itermax):# On calcule 99 barycentres si on a 100 sujets
        if X is None:
            # On prend les 2 premiers sujets de la liste position 0 et 1
            X = ot.lp.free_support_barycenter(measures_locations[:2], measures_weights[:2],X_init,b)
        else:
            X_init=X
            L_loc=[X_init,measures_locations[i+itermax]]#sujet suivant pour calcul du barycentre position 2
            L_w=[b,measures_weights[i+itermax]]
            X = ot.lp.free_support_barycenter(L_loc, L_w,X_init,b,weights=np.array([(i+itermax)/(i+itermax+1),1/(i+itermax+1)]),numItermax=100)  
        # Sauvegarde
        tools.save_value(X,str(i+itermax),directory=destination)
        display.show_dot(X,title='Barycenter')
        tools.save_fig(str(i+itermax),directory=destination)
"""

"""
# Calcule des distances par rapport au centroide
if True:
    numpy_vars = {}
    for np_name in glob.glob('exp2/'+'*.np[yz]'):
        numpy_vars[np_name] = np.load(np_name)
        
    for X in numpy_vars:
        L=[]
        X=np.load(X)
        #X=np.load(destination+"99.npy")
        #X=np.load('.'+str(np.load(variables+'centroide.npy')).replace('\\','/'))
        for i in numpy_vars:
            M=ot.dist(X,np.load(i))
            G=ot.emd(ot.unif(len(X)),measures_weights[i],M,numItermax=1000000)
            cost=G*M
            L.append(np.sum(cost))
        #tools.save_value(L,'L',destination)    
        #tools.save_value(np.sum(L),'L_sum',destination)
        print(np.sum(L))
"""
"""
#Déterminer les maximums locaux par le code d'alex
from sklearn.cluster import DBSCAN

# Parameters obtained and fixed for DBSCAN clustering of group profiles
EPS = 3
ABS = 12000
NORM_THRESHOLD = ABS / 249897.0
print(NORM_THRESHOLD)

def estimate_weights(coordinates):
    unique_coords, inverse, weigth = np.unique(coordinates, return_inverse=True, return_counts=True, axis=0)
    return unique_coords, inverse, weigth


def dbscan_density_clustering(data, eps=EPS, normalised_threshold=NORM_THRESHOLD):
    unique_data, inverse, weigth = estimate_weights(data)
    total_weigth = np.sum(weigth)
    # normalisation to work with densities
    min_sample = total_weigth * normalised_threshold
    db = DBSCAN(eps=eps, min_samples=min_sample).fit(X=unique_data, sample_weight=weigth)
    unique_labels = db.labels_
    labels = unique_labels[inverse]
    return labels


# Chemins
source='../data/L/'
variables='../variables/L/'
destination='test/L/'

if False:
    import glob
    Xsum=np.load('.'+str(np.load(variables+'centroide.npy')).replace('\\','/'))
    _,_,Xsum=tools.estimate_pseudo_density(Xsum)
    Xsum=Xsum*0
    
    numpy_vars = {}
    for np_name in glob.glob(source+'*.np[yz]'):
        _,_,X=tools.estimate_pseudo_density(np.load(np_name))
        Xsum=Xsum+X
    display.show_map(Xsum, "Group profile registered")

data=np.load('test/L/Xsum.npy')
data=data/np.sum(data)
X=np.load('.'+str(np.load(variables+'centroide.npy')).replace('\\','/'))
_,_,data=tools.estimate_pseudo_density(X)

labels=dbscan_density_clustering(data)
print(np.sum(labels))
"""

"""
#Information sur les données
# Chemins
source='../data/L/'
variables='../variables/L/'
destination='test'
size=len(source)-1

# Changement des données
measures_locations = []
measures_weights = []
L_name=np.load(str(variables)+'L_name.npy')
L_val=np.load(str(variables)+'L_val.npy')
L_trie=[L_name[i][size:] for i in np.argsort(L_val)]
#L_trie=[L_name[i][size:] for i in range(np.shape(L_val)[0])]#minrandom
# pour 10 sujets
i=0
for np_name in  L_trie:#glob.glob(str(source)+'*.np[yz]'):
    np_name=source+np_name
    measures_locations.append(np.load(np_name))
    if i>100:
        break
    i=i+1

#Min  
print(int(np.min([np.shape(i)[0] for i in measures_locations])))
var=L_trie[int(np.argmin([np.shape(i)[0] for i in measures_locations]))]
title=var[:-4]
X=np.load(source+var)
display.show_dot(X,title=title)
tools.save_fig(title,destination)

_,_,XX=tools.estimate_pseudo_density(X)
display.show_map(XX,title=title)
tools.save_fig('map_'+title,destination)
print(title)

#Max
var=L_trie[int(np.argmax([np.shape(i)[0] for i in measures_locations]))]
title=var[:-4]
X=np.load(source+var)
display.show_dot(X,title=title)
tools.save_fig(title,destination)

_,_,XX=tools.estimate_pseudo_density(X)
display.show_map(XX,title=title)
tools.save_fig('map_'+title,destination)
print(title)

# Median
a=np.shape(measures_locations[np.argsort([np.shape(i)[0] for i in measures_locations])[49]])[0] # median)
b=np.shape(measures_locations[np.argsort([np.shape(i)[0] for i in measures_locations])[50]])[0] # median)
print(int((a+b)/2))
var=L_trie[np.argsort([np.shape(i)[0] for i in measures_locations])[49]]
title=var[:-4]
X=np.load(source+var)
display.show_dot(X,title=title)
tools.save_fig(title,destination)

_,_,XX=tools.estimate_pseudo_density(X)
display.show_map(XX,title=title)
tools.save_fig('map_'+title,destination)
print(title)

var=L_trie[np.argsort([np.shape(i)[0] for i in measures_locations])[50]]
title=var[:-4]
X=np.load(source+var)
display.show_dot(X,title=title)
tools.save_fig(title,destination)

_,_,XX=tools.estimate_pseudo_density(X)
display.show_map(XX,title=title)
tools.save_fig('map_'+title,destination)
print(title)

# Moments statistiques
print(int(np.mean([np.shape(i)[0] for i in measures_locations])))
print(np.std([np.shape(i)[0] for i in measures_locations]))

from scipy.stats import kurtosis as kurtosis
print(kurtosis([np.shape(i)[0] for i in measures_locations]))
"""


"""
#local maximum
from skimage import morphology as skm
mask1=np.array([[0,1,0],
               [1,1,1],
               [0,1,0]])


mask=np.full((3,3),1)
mask=np.full((5,5),1)
#mask=skm.square(5)

print(mask)
#X=local_maxima(X,selem=mask)
#display.plot_map2(X,r,sub1title='Moyenne',sub2title='Maximum')

####
#r=local_maxima(data,selem=mask)
#display.plot_map2(data,r,sub1title='Barycentre',sub2title='Maximum')
"""

"""
#annexe centroide
val_W22=[np.around(i,2) for i in np.load("../variables/R/L_val.npy")]
argval_W22=np.argsort(val_W22)
val_W22=[val_W22[i] for i in argval_W22]

val_W2=[np.around(np.sqrt(i/99),2) for i in val_W22]

name=[i[9:17] for i in np.load("../variables/R/L_name.npy")]
name=[name[i] for i in argval_W22]

point=[np.shape(np.load('../data/R/'+str(i)+'_connectivity_withHKnob.npy'))[0] for i in name]
"""
import matplotlib.pylab as plt
data=np.load('../data/L/495255_L_connectivity_withHKnob.npy')
_,_,data=tools.estimate_pseudo_density(data)
grid_size=100

plt.figure()
extent = (0,grid_size-1 , 0,grid_size-1)
plt.imshow(data,cmap=plt.cm.magma_r,origin='lower',extent=extent)
plt.autoscale(False)
plt.axis('on')
plt.xlabel('Precentral gyral crest scaled to 100')
plt.ylabel('Postcentral gyral crest scaled to 100')
plt.colorbar()
plt.grid(linestyle = '--', linewidth = 0.5,alpha=0.5, which='major')
plt.title('L - hemisphere')
tools.save_fig('L','./test')

sys.exit()