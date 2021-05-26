# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 30/03/2021
@version: 1.80
@revision: 26/05/2021
@Recommandation: Python 3.7

@list of functions 
- show_dot(xs,xs_color=None,title=None,size=(10,10))
- show_map(xs,title,size=(10,10))

- plot_dots(xs,xt,xs_color=None,xt_color=None,title=None,size=(10,10))
- plot_dots2(xs,xt,xs_color=None,xt_color=None,size=(20,20))

- link(xs, xt, pos,weight=None,color_weight=0.2, **kwargs)
- label(xs, xt, pos, color)

- plot_dots_projection(xs,xt,pos,xs_color=None,xt_color=None,size=(60,20))
- plot_dots_links(xs,xt,pos,weight=None,xs_color=None,title=None,size=(20,20))

"""
import matplotlib.pylab as plt
import numpy as np

def link(xs, xt, pos,weight=None,color_weight=0.2, **kwargs):
    """ Labelling target samples depending on source samples contribution
    
    Plot labelling color in target space depending on the label of source sample

    Parameters
    ----------
    xs : ndarray, shape (ns,2)
        Source samples positions
    xt : ndarray, shape (nt,2)
        Target samples positions
    pos : ndarray, shape (nt,2)
        Correspondance xt dots with ns dots. We don't nessesary have bijection between ns and nt
    weight : ndarray, shape (nt,2)
        weight of bridge between xt dots with ns dots. We don't nessesary have bijection between ns and nt        
    color_weight : float, optional
        Color of links. The default is None
    **kwargs : dict
        parameters given to the plot functions (default color is black if
        nothing given)

    Returns
    -------
    None.

    """
    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'
    if weight is None:
        for i in pos:
            plt.plot([xs[i[1], 0], xt[i[0], 0]], [xs[i[1], 1], xt[i[0], 1]],alpha=color_weight, **kwargs)
    else:
        max=np.max(weight[:,1])
        for i in list(zip(pos,weight)):
            plt.plot([xs[i[0][1], 0], xt[i[0][0], 0]], [xs[i[0][1], 1], xt[i[0][0], 1]], alpha=i[1][1]/max, **kwargs)
  
def label(xs, xt, pos, color):
    """ Labelling target samples depending on source samples contribution
    
    Plot labelling color in target space depending on the label of source sample

    Parameters
    ----------
    xs : ndarray, shape (ns,2)
        Source samples positions
    xt : ndarray, shape (nt,2)
        Target samples positions
    pos : ndarray, shape (nt,2)
        Correspondance xt dots with ns dots. We don't nessesary have bijection between ns and nt
    color : list, shape (ns,3)
            Source color label in RGB (r,g,b)

    Returns
    -------
    None.

    """
    cols_bis=[]
    for i in pos:
        cols_bis.append([i[0],color[i[1]]])
    for i in cols_bis:
        plt.scatter(xt[i[0],0],xt[i[0],1],marker = '*', s = 50, color = i[1])

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
        
def plot_dots_projection(xs,xt,pos,xs_color=None,xt_color=None,size=(60,20)):
    """Plot 3 figures with labellisation
    
    Generate 3 figures with source dots, target dots and an other with source projected
    

    Parameters
    ----------
    xs : ndarray, shape (ns,2)
        Source samples positions
    xt : ndarray, shape (nt,2)
        Target samples positions
    pos : ndarray, shape (nt,2)
        Correspondance xt dots with ns dots. We don't nessesary have bijection between ns and nt
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
    label(xs, xt, pos, xs_color)
    plt.title('Source projected')
        
def plot_dots_links(xs,xt,pos,weight=None,xs_color=None,title=None,size=(20,20)):
    """Plot 1 figure with links
    
    Generate 1 figure with source dots and target dots, links between and labbeling of source dots in target dots space.
    

    Parameters
    ----------
    xs : ndarray, shape (ns,2)
        Source samples positions
    xt : ndarray, shape (nt,2)
        Target samples positions
    pos : ndarray, shape (nt,2)
        Correspondance xt dots with ns dots. We don't nessesary have bijection between ns and nt
    weight : ndarray, shape (nt,2)
        weight of bridge between xt dots with ns dots. We don't nessesary have bijection between ns and nt        
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
    link(xs, xt, pos,weight,c=[0.5,0.5,0.5])
    label(xs, xt, pos,xs_color)
    plt.scatter(xs[:,0], xs[:,1],marker = '+', s = 50)
    
    ns=xs.shape[0]
    nt=xt.shape[0]
    if title is None:
        title='Source ({ns}), Source projected ({nt})'.format(ns=ns,nt=nt)
        plt.title(title)
    else:
        plt.title(title)        

def show_dot(xs,xs_color=None,title=None,size=(10,10)):
    plt.figure(figsize=size)
    plt.xlim(0,100) 
    plt.ylim(0,100)
    plt.scatter(xs[:,0], xs[:,1],marker = '+', s = 50, color = xs_color)
    plt.title(title)
    
def show_map(xs,title,size=(10,10),cmap=plt.cm.magma_r,origin='lower'):
    if origin=='lower':
        extent=(0, 100, 0, 100)
    else:
        extent=(0, 100, 100,0)
        
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title)
    ax.imshow(xs,cmap,origin=origin,extent=extent)
    
    
def plot_map2(x1,x2,sub1title=None,sub2title=None,size=(10,20),cmap=plt.cm.magma_r,origin='lower'):
    if origin=='lower':
        extent=(0, 100, 0, 100)
    else:
        extent=(0, 100, 100,0)
        
    fig, axs = plt.subplots(1,2,figsize=size)
    axs[0].set_title(sub1title)
    axs[0].imshow(x1,cmap,origin=origin,extent=extent)
    axs[1].set_title(sub2title)
    axs[1].imshow(x2,cmap,origin=origin,extent=extent)