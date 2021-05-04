# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 04/05/2021
@version: 1.25
@Recommandation: Python 3.7
@But: comparaison des algorithmes de barycentre
"""
import numpy as np
import sys
import glob
import ot


from time import process_time

sys.path.insert(1,'../libs')
import tools, display

# calcul barycentre 
def free_support_barycenter(measures_locations, measures_weights, X_init, b=None, weights=None, numItermax=1000, stopThr=1e-7,num2Itermax=102400):
    """
    Take from : ot.lp.free_support_barycenter
    
    Solves the free support (locations of the barycenters are optimized, not the weights) Wasserstein barycenter problem (i.e. the weighted Frechet mean for the 2-Wasserstein distance)

    The function solves the Wasserstein barycenter problem when the barycenter measure is constrained to be supported on k atoms.
    This problem is considered in [1] (Algorithm 2). There are two differences with the following codes:
    - we do not optimize over the weights
    - we do not do line search for the locations updates, we use i.e. theta = 1 in [1] (Algorithm 2). This can be seen as a discrete implementation of the fixed-point algorithm of [2] proposed in the continuous setting.
    

    Parameters
    ----------
    measures_locations : list of (k_i,d) numpy.ndarray
        The discrete support of a measure supported on k_i locations of a d-dimensional space (k_i can be different for each element of the list)
    measures_weights : list of (k_i,) numpy.ndarray
        Numpy arrays where each numpy array has k_i non-negatives values summing to one representing the weights of each discrete input measure

    X_init : (k,d) np.ndarray
        Initialization of the support locations (on k atoms) of the barycenter
    b : (k,) np.ndarray
        Initialization of the weights of the barycenter (non-negatives, sum to 1)
    weights : (k,) np.ndarray
        Initialization of the coefficients of the barycenter (non-negatives, sum to 1)
        
    numItermax : int, optional
        Max number of iterations. The default is 1000.
    stopThr : float, optional
        Stop threshold on error (>0). The default is 1e-7.
    num2Itermax : int, optional
        Max number of iterations for emd. The default is 102400.

    Returns
    -------
    X : (k,d) np.ndarray
        Support locations (on k atoms) of the barycenter.

    """
    N = len(measures_locations)
    k,d = X_init.shape
    
    if weights is None:
        weights = np.ones((N,)) / N

    displacement_norm = stopThr + 1.
    iter_count = 0
    
    X = X_init
    result_code=1
    while (displacement_norm > stopThr and iter_count < numItermax):

        T_sum = np.zeros((k, d))

        for (measure_locations_i, measure_weights_i, weight_i) in zip(measures_locations, measures_weights,weights.tolist()):
            M_i = ot.dist(X, measure_locations_i)
            T_i,_, _, _, result_code = ot.lp.emd_c(b, measure_weights_i, M_i,num2Itermax)
            #T_i = ot.bregman.sinkhorn(b, measure_weights_i, M_i,reg=10,numItermax=100000) # résolution plus lente
            if result_code!=1:
                #print('EMD old max iteration : '+str(num2Itermax))
                num2Itermax=num2Itermax*2
                print('EMD new max iteration : '+str(num2Itermax))
            T_sum = T_sum + weight_i * np.reshape(1. / b, (-1, 1)) * np.matmul(T_i, measure_locations_i)
        displacement_norm = np.sqrt((np.sum(np.square(T_sum - X))))
        #print('Barycentre : '+str(displacement_norm))
        X = T_sum
        iter_count += 1
    return X

# calcul barycentre sujet itératif
def iterative_barycenter(X,X_init,b,measures_locations,measures_weights,Nmax=100,itermax=0,stopThr=0.01,num2Itermax=102400,destination='temp/'):
    """ Compute barycenter profil by profil
    

    Parameters
    ----------
    X : (k,d) np.ndarray
        Support locations (on k atoms) of the barycenter.
    X_init : (k,d) np.ndarray
        Initialization of the support locations (on k atoms) of the barycenter
    b : (k,) np.ndarray
        Initialization of the weights of the barycenter (non-negatives, sum to 1)
        
    measures_locations : list of (k_i,d) numpy.ndarray
        The discrete support of a measure supported on k_i locations of a d-dimensional space (k_i can be different for each element of the list)
    measures_weights : list of (k_i,) numpy.ndarray
        Numpy arrays where each numpy array has k_i non-negatives values summing to one representing the weights of each discrete input measure

    Nmax : int, optional
        Number of profil. The default is 100.
    itermax : int, optional
        Profil already computed. The default is 0.
        
    stopThr : float, optional
        Stop threshold on error (>0). The default is 0.01.
    num2Itermax : int, optional
        Max number of iterations for emd. The default is 102400.
        
    destination : TYPE, optional
        Directory of temp profil computed. The default is 'temp/'.

    Returns
    -------
    None.

    """
    for i in range(1,Nmax-itermax):# On calcule 99 barycentres si on a 100 sujets
        if X is None:
            # On prend les 2 premiers sujets de la liste position 0 et 1
            X = free_support_barycenter(measures_locations[:2], measures_weights[:2],X_init,b,stopThr=stopThr,num2Itermax=num2Itermax)
        else:
            X_init=X
            L_loc=[X_init,measures_locations[i+itermax]]#sujet suivant pour calcul du barycentre position 2
            L_w=[b,measures_weights[i+itermax]]
            X = free_support_barycenter(L_loc, L_w,X_init,b,weights=np.array([(i+itermax)/(i+itermax+1),1/(i+itermax+1)]),stopThr=stopThr,num2Itermax=num2Itermax)  
        # Sauvegarde
        tools.save_value(X,str(i+itermax),directory=destination)
        
# Chemins
source='../data/R/'
variables='../variables/R/'
destination='barycentre_R/minbis/'

size=len(source)-1

# Changement des données
measures_locations = []
measures_weights = []
L_name=np.load(str(variables)+'L_name.npy')
L_val=np.load(str(variables)+'L_val.npy')
L_trie=[L_name[i][size:] for i in np.argsort(L_val)]

# pour 10 sujets
i=0
for np_name in  L_trie:#glob.glob(str(source)+'*.np[yz]'):
    np_name=source+np_name
    measures_locations.append(np.load(np_name))
    measures_weights.append(ot.unif(len(measures_locations[-1])))
    if i>100:
        break
    i=i+1
Nmax=np.shape(measures_locations)[0]

# Initialisation du profil type

if False:# Prend du temps     
    #Calcul du barycentre itératif
    itermax=0
    # k=int(np.max([np.shape(i)[0] for i in measures_locations]))
    #k = int(np.mean([np.shape(i)[0] for i in measures_locations]))# number of Diracs of the barycenter
    # X_init = np.random.normal(0., 1., (k, 2))  # initial Dirac locations
    # b = np.ones((k,)) / k  # weights of the barycenter (it will not be optimized, only the locations are optimized)
    
    k=np.argmin([np.shape(i)[0] for i in measures_locations[:-1]]) #max
    #k=np.argsort([np.shape(i)[0] for i in measures_locations])[len(measures_locations)//2] # median
    #k=np.argmax([np.shape(i)[0] for i in measures_locations]) #max
    #k=-1 # profile le moins représentatif par rapport au calcul du centroide = min
    X_init = measures_locations[k]
    
    #X_init=np.load('.'+str(np.load(variables+'centroide.npy')).replace('\\','/'))  # centroide
    b=ot.unif(np.shape(X_init)[0])

    X=None
    t1_start = process_time() 
    iterative_barycenter(X,X_init,b,measures_locations,measures_weights,Nmax,itermax,stopThr=0.01,destination=destination)    
    t1_stop = process_time() 
    print(t1_stop-t1_start)
    
    # Sauvegarde des images
    for np_name in glob.glob(str(destination)+'*.np[yz]'):
        display.show_dot(np.load(np_name),title='Barycenter')
        tools.save_fig('dot_'+np_name.replace('\\','/')[len(destination):-4],directory=destination+'Dot')
    
        _,_,Img_xs=tools.estimate_pseudo_density(np.load(np_name))
        display.show_map(Img_xs,title='Barycenter')
        tools.save_fig('map_'+np_name.replace('\\','/')[len(destination):-4],directory=destination+'Map')
        
if False:
    L=[]
    X=np.load("barycentre_R/centroide/99.npy")
    X=np.load('.'+str(np.load(variables+'centroide.npy')).replace('\\','/'))
    for i in range(np.shape(measures_locations)[0]):
        M=ot.dist(X,measures_locations[i])
        G=ot.emd(ot.unif(len(X)),measures_weights[i],M,numItermax=1000000)
        cost=G*M
        L.append(np.sum(cost))
    tools.save_value(L,'L','barycentre_R/ref/')    
sys.exit()