# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 30/03/2021
@version: 1.4
@revision: 12/04/2021
@Recommandation: Python 3.7
"""
import numpy as np
import matplotlib.pylab as plt

from operator import itemgetter
import os
import ot
from scipy.stats import gaussian_kde

def data_generator_simulation1():
  """ Generate source samples and target samples

  Generate 2 ellipsoide clouds of dot for the source samples 
  and 1 ellipsoide cloud of dat for the target samples with a uniforme contribution for each dot.
  """
  # Target : 1 nuage de point
  nt = 1000
  mu_t = np.array([50, 50])
  cov_t = np.array([[60, 40], 
                    [40, 60]])
  Xt = ot.datasets.make_2D_samples_gauss(nt, mu_t, cov_t)

  # Source : 3 nuages de points
  ns1 = 700
  mu_s = np.array([25, 60])
  cov_s = np.array([[30, 10], 
                    [10, 30]])
  Xs = ot.datasets.make_2D_samples_gauss(ns1, mu_s, cov_s)

  ns2 = 400
  mu_s = np.array([55, 80])
  cov_s = np.array([[30, 10], 
                    [10, 30]])
  Xs=np.append(Xs,ot.datasets.make_2D_samples_gauss(ns2, mu_s, cov_s),axis=0)


  # Compute the distribution laws associate with the clouds of dots.
  ns=ns1+ns2
  a, b = ot.unif(ns), ot.unif(nt)  # uniform distribution on samples
  return (Xs,a),(Xt,b)

def data_generator_simulation2(rand_seed=0,noise=0.1):
    """Generate data simulate real streamline


    Parameters
    ----------
    rand_seed : TYPE, optional
        Number of noise seed. The default is 0.
    noise : TYPE, optional
        Ratio of noise in the data. The default is 0.1.

    Returns
    -------
    Xs : ndarray, shape (ns,2)
        Source samples positions
    a : ndarray, shape (ns,2)
        Source samples amplitudes

    """
    temp=0
    
    # Ventral : bouche
    ns = 120
    temp=temp+ns
    mu_s = np.array([9, 9])
    cov_s = np.array([[10,0], 
                      [0, 10]])
    Xs = ot.datasets.make_2D_samples_gauss(ns, mu_s, cov_s)
    
    # Mediane : main
    ns = 40
    temp=temp+ns
    mu_s = np.array([55, 50])
    cov_s = np.array([[2, 0], 
                      [0, 10]])
    Xs=np.append(Xs,ot.datasets.make_2D_samples_gauss(ns, mu_s, cov_s),axis=0)
    
    ns = 120
    temp=temp+ns
    mu_s = np.array([65, 65])
    cov_s = np.array([[5,5], 
                      [5, 15]])
    Xs=np.append(Xs,ot.datasets.make_2D_samples_gauss(ns, mu_s, cov_s),axis=0)
    
    ns = 80
    temp=temp+ns
    mu_s = np.array([75, 75])
    cov_s = np.array([[10, 0], 
                      [0, 10]])
    Xs=np.append(Xs,ot.datasets.make_2D_samples_gauss(ns, mu_s, cov_s),axis=0)            
    
    #Dorsale : pied
    ns = 120
    temp=temp+ns
    mu_s = np.array([95, 90])
    cov_s = np.array([[2, 0], 
                      [0, 20]])
    Xs=np.append(Xs,ot.datasets.make_2D_samples_gauss(ns, mu_s, cov_s),axis=0)
    
    # Noise
    if rand_seed!=0:
        ns_noise=int(noise*ns/rand_seed)
        for i in range(rand_seed):
            mu_s = np.array([np.random.randint(0,100), np.random.randint(0,100)])
            cov_s = np.array([[np.random.randint(1,20),np.random.randint(0,2)], 
                              [np.random.randint(0,2), np.random.randint(1,10)]])
            Xs=np.append(Xs,ot.datasets.make_2D_samples_gauss(ns_noise, mu_s, cov_s),axis=0)
        temp=temp+ns_noise
    # Compute the distribution laws associate with the clouds of dots.
    a = ot.unif(temp) # uniform distribution on samples
    return (Xs,a)

def extract_point(S):
  """Extract from image a list of dot
  
  
  Parameters
  ----------
  S : ndarray
      Carte de connectivité
  Returns
  -------
  Tuple of 2 elements containing:
    Lcoord : numpy.ndarray
            Liste des coordonnées des valeurs non nulles
    Lampl : numpy.ndarray
            Liste des amplitudes associées aux coordonnées normalisées sous la forme de (y,x)      
      
  """  
  h,w=np.shape(S)
  Lcoord=[]
  Lampl=[]
  
  for i in range(h):
    for j in range(w):
      if S[i,j]>0:
        Lampl.append(S[i,j])
        Lcoord.append(np.array([j,i]))
  Lampl=Lampl/sum(Lampl)#normalisation
  
  return np.array(Lcoord),np.array(Lampl)

def label_amplitude(a,b):
    """Amplitude labbeling
    

    Parameters
    ----------
    a : ndarray, shape (ns,2)
        Source samples amplitudes
    b : ndarray, shape (nt,2)
        Target samples amplitudes

    Returns
    -------
    xs_color : list of tuple (R,G,B), optional
        list of label of source. The default is None.
    xt_color : list of tuple (R,G,B), optional
        list of label of target. The default is None.

    """
    #Label généré en fonction de l'amplitude
    xs_color=[ (0,1-i,i) for i in a/np.max(a)]
    xt_color=[ (i,0,1-i) for i in b/np.max(b)]
    
    return (xs_color,xt_color)

def label_position(xs,xt):
    """Position labbeling
    

    Parameters
    ----------
    xs : ndarray, shape (ns,2)
        Source samples positions
    xt : ndarray, shape (nt,2)
        Target samples positions

    Returns
    -------
    xs_color : list of tuple (R,G,B), optional
        list of label of source. The default is None.
    xt_color : list of tuple (R,G,B), optional
        list of label of target. The default is None.

    """
    ns=xs.shape[0]
    nt=xt.shape[0]
    
    x=xs[:,0].argsort()
    y=xs[:,1].argsort()
    xs_color = [(i[0]/ns,1-i[1]/ns,0) for i in list(zip(x,y))]
    
    x=xt[:,0].argsort()
    y=xt[:,1].argsort()
    xt_color = [(0,i[0]/nt,1-i[1]/nt) for i in list(zip(x,y))]
    
    return (xs_color,xt_color)

def degree(xs,xt,G,degree=1):
    """
    

    Parameters
    ----------
    xs : ndarray, shape (ns,2)
        Source samples positions
    xt : ndarray, shape (nt,2)
        Target samples positions
    G : ndarray, shape (na,nb)
        OT matrix
    degree : int, optional
        Degree of contribution of source samples for target samples depending on the weight.
        The default is 1

    Returns
    -------
    pos : list, shape (nt,1)
        source projected for a degree
    weight : list, shape (nt,1)
        weight of projection for a degree

    """
    weight=[]
    pos=[]
    for j in range(xt.shape[0]):
        # Initialisation
        temp=[]
        temp1=degree
      
        # Controle
        for i in range(xs.shape[0]):
          if G[i,j]>0:
            temp.append([G[i, j],i])
      
        # Classement
        temp=sorted(temp, key=itemgetter(0))#trie selon le poid dans l'ordre croissant
        temp=temp[::-1]
        # Labelisation
        if len(temp)!=0:
            if (degree) > len(temp):
                temp1=len(temp)
            weight.append([j,temp[temp1-1][0]])
            pos.append([j,temp[temp1-1][1]])#point de xs liée à xt (on a l'ensemble des xt qui a une correspondance vers xs)
    return (np.array(pos),np.array(weight))
        
def save_value(value,title,directory=''):
    """Save value
    

    Parameters
    ----------
    value : 
        value to save.
    title : string
        name of the variable.
    directory : string, optional
        Directory of the file. The default is ''.

    Returns
    -------
    None.

    """
    if not os.path.exists(directory):
        os.makedirs(directory) 
    temp=str(directory)+'/'+str(title)+'.npy'
    with open(temp,'wb') as f:
        np.save(f,value)

def save_fig(title,directory=''):
    """Save figure
    

    Parameters
    ----------
    title : string
        name of the figure.
    directory : TYPE, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    None.

    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    temp=str(directory)+'/'+str(title)+'.png'
    plt.savefig(temp,format="png", transparent=True)
    plt.close()

def unnormalized_kernel(points, factor=10):
    """
    Modify the behavior of the gaussian_kde object to
    enforce a fixed size kernel (spherical gaussian)
    This kernel is not normalized for the number of points
    and the unit integration over the domain so that values
    estimated are close to the initial. "Density map" or function
    estimated using this kernel can be compared between each other
    :param points: (N,D) ndarray, N the number of points and D the dimension of the space
    :return: A kernel object
    """
    kernel = gaussian_kde(points.T, bw_method=1)
    # random spherical gaussian no prior
    kernel._data_covariance = factor * np.eye(points.T.shape[0])
    kernel.covariance = kernel._data_covariance * kernel.factor ** 2
    kernel._data_inv_cov = np.linalg.inv(kernel._data_covariance)
    kernel.covariance = kernel._data_covariance * kernel.factor ** 2
    kernel.inv_cov = kernel._data_inv_cov / kernel.factor ** 2
    # kernel._norm_factor = np.sqrt(linalg.det(2*np.pi*kernel.covariance))
    kernel._norm_factor = 1
    return kernel

def estimate_pseudo_density(points, grid_size=101, factor=10):
    """
    Estimate the (unormalized connectivity_profiles with fixed kernel size)
    :param points:
    :param grid_size:
    :return:
    """
    kernel = unnormalized_kernel(points, factor)
    # creation of a grid to display the function
    x = y = np.linspace(0, 100, num=grid_size)
    X, Y = np.meshgrid(x, y)
    new_points = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(new_points), X.shape)
    return X, Y, Z

def continue2discret(Img_Xs,Img_Xt,seuil=10e-4):
    # Normalisation 1
    Img_Xs=Img_Xs/np.max(Img_Xs)
    Img_Xt=Img_Xt/np.max(Img_Xt)
    
    # Seuillage on ejecte 1% des valeurs les plus petites
    Img_Xs=(Img_Xs>seuil)*Img_Xs
    Img_Xt=(Img_Xt>seuil)*Img_Xt
    
    # Normalisation 2
    Img_Xs=Img_Xs/np.max(Img_Xs)
    Img_Xt=Img_Xt/np.max(Img_Xt)
    return (Img_Xs,Img_Xt)