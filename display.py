# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 30/03/2021
@version: 1.0
@Recommandation: Python 3.7
"""
import matplotlib.pylab as plt
from operator import itemgetter

def link(xs, xt, G, degree=1,color_weight=None, **kwargs):
    """ Plot matrix M in 2D with lines using alpha values
    
    Plot lines between source and target 2D samples with a color
    proportional to the value of the matrix G between samples at a degree of contribution.
    
    
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
    color_weight : float, optional
        Color of links. The default is None
        
    **kwargs : dict
        parameters given to the plot functions (default color is black if
        nothing given)
        
    """
    
    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'
    
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
            
            # Affichage des liens au degrés indiqué.
            ii=temp[temp1-1]
            if color_weight is None:
              val=G[i, j]/G.max()
            else:
              val=color_weight
            plt.plot([xs[ii[1], 0], xt[j, 0]], [xs[ii[1], 1], xt[j, 1]], alpha=val, **kwargs)

def label(xs, xt, G, color,degree=1):
    """ Labelling target samples depending on source samples contribution
    
    Plot labelling color in target space depending on the label of source sample
    
    Parameters
    ----------
    xs : ndarray, shape (ns,2)
        Source samples positions
    xt : ndarray, shape (nt,2)
        Target samples positions
    G : ndarray, shape (na,nb)
        OT matrix
    color : list, shape (ns,3)
            Source color label in RGB (r,g,b)
    degree : int, optional
        Degree of contribution of source samples for target samples depending on the weight.
        The default is 1
        
    """
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
          plt.scatter(xt[j,0], xt[j,1],marker = 'p', s = 50, color = color[temp[temp1-1][1]])

def plot_dots(xs,xt,xs_color=None,xt_color=None,title=None,size=(10,10)):
    """ Plot 1 figure
    
    Generate 1 figure with source dots and target dots


    Parameters
    ----------
    xs : ndarray, shape (ns,2)
        Source samples positions
    xt : ndarray, shape (nt,2)
        Target samples positions.
    xs_color : list of tuple (R,G,B), optional
        list of label of source. The default is None.
    xt_color : list of tuple (R,G,B), optional
        list of label of target. The default is None.
    title : string, optional
        title of figure. The default is None.
    size : tuple, optional
        size of figure. The default is (10,10).

    Returns
    -------
    None.

    """
    plt.figure(figsize=size)
    plt.xlim(0,100) 
    plt.ylim(0,100)
    plt.scatter(xs[:,0], xs[:,1],marker = '+', s = 50, color = xs_color)
    plt.scatter(xt[:,0], xt[:,1],marker = 'p', s = 50, color= xt_color)
    
    ns=xs.shape[0]
    nt=xt.shape[0]
    if title is None:
        title='Source ({ns}), Source projected ({nt})'.format(ns=ns,nt=nt)
        plt.title(title)
    else:
        plt.title(title)
        
def plot_dots2(xs,xt,xs_color=None,xt_color=None,size=(20,20)):
    """Plot 2 figures
    
    Generate 2 figures with source dots and an other with target dots
    

    Parameters
    ----------
    xs : ndarray, shape (ns,2)
        Source samples positions
    xt : ndarray, shape (nt,2)
        Target samples positions
    xs_color : list of tuple (R,G,B), optional
        list of label of source. The default is None.
    xt_color : list of tuple (R,G,B), optional
        list of label of target. The default is None.
    size : tuple, optional
        size of figure. The default is (20,20).

    Returns
    -------
    None.

    """
    fig=plt.figure(figsize=size)
    
    fig.add_subplot(2,2,1)
    plt.xlim(0,100) 
    plt.ylim(0,100)
    plt.scatter(xs[:,0], xs[:,1],marker = '+', s = 100, color = xs_color)
    plt.title('Source')
    
    fig.add_subplot(2,2,2)
    plt.xlim(0,100) 
    plt.ylim(0,100)
    plt.scatter(xt[:,0], xt[:,1],marker = 'p', s = 100, color = xt_color)
    plt.title('Cible')
        
def plot_dots_projection(xs,xt,G,xs_color=None,xt_color=None,size=(60,20)):
    """Plot 3 figures with labellisation
    
    Generate 3 figures with source dots, target dots and an other with source projected
    

    Parameters
    ----------
    xs : ndarray, shape (ns,2)
        Source samples positions
    xt : ndarray, shape (nt,2)
        Target samples positions
    G : ndarray, shape (na,nb)
        OT matrix
    xs_color : list of tuple (R,G,B), optional
        list of label of source. The default is None.
    xt_color : list of tuple (R,G,B), optional
        list of label of target. The default is None.
    size : tuple, optional
        size of figure. The default is (60,20).

    Returns
    -------
    None.

    """
    fig=plt.figure(figsize=size)
    
    fig.add_subplot(1,3,1)
    plt.xlim(0,100) 
    plt.ylim(0,100)
    plt.title('Source')
    plt.scatter(xs[:,0], xs[:,1],marker = '+', s = 100, color = xs_color)
    
    fig.add_subplot(1,3,2)
    plt.xlim(0,100) 
    plt.ylim(0,100)
    plt.title('Cible')
    plt.scatter(xt[:,0], xt[:,1],marker = 'p', s = 100, color = xt_color)
    
    fig.add_subplot(1,3,3)
    plt.xlim(0,100) 
    plt.ylim(0,100)
    label(xt, xt, G,xs_color)
    plt.title('Source projected')
        
def plot_dots_links(xs,xt,G,degree_link=1,degree_label=1,color_weight=0.2,xs_color=None,title=None,size=(20,20)):
    """Plot 1 figure with links
    
    Generate 1 figure with source dots and target dots, links between and labbeling of source dots in target dots space.
    

    Parameters
    ----------
    xs : ndarray, shape (ns,2)
        Source samples positions
    xt : ndarray, shape (nt,2)
        Target samples positions
    G : ndarray, shape (na,nb)
        OT matrix
    degree_link : int, optional
        Degree of contribution of source samples for target samples depending on the weight by links.
        The default is 1
    degree_label : int, optional
        Degree of contribution of source samples for target samples depending on the weight by labels.
        The default is 1
    color_weight :  float, optional
        Color of links. The default is 0.2.
    xs_color : list of tuple (R,G,B), optional
        list of label of source. The default is None.
    title : string, optional
        title of figure. The default is None.
    size : tuple, optional
        size of figure. The default is (20,20).

    Returns
    -------
    None.

    """
    plt.figure(figsize=size)
    plt.xlim(0,100) 
    plt.ylim(0,100)
    link(xs, xt, G,degree_link,color_weight,c=[0.5,0.5,0.5])
    label(xs, xt, G,xs_color,degree=degree_label)
    plt.scatter(xs[:,0], xs[:,1],marker = '+', s = 50, color = xs_color)
    
    ns=xs.shape[0]
    nt=xt.shape[0]
    if title is None:
        title='Source ({ns}), Source projected ({nt})'.format(ns=ns,nt=nt)
        plt.title(title)
    else:
        plt.title(title)        
