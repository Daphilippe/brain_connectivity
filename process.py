# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 30/03/2021
@version: 1.0
@Recommandation: Python 3.7
"""
import display
import tools

import matplotlib.pylab as plt

def auto_affichage_discret(Xs,Xt,M,G,serie):
    """ Automatisation display processus for dots

    Parameters
    ----------
    Xs : ndarray, shape (ns,2)
        Source samples positions
    Xt : ndarray, shape (nt,2)
        Target samples positions
    M : ndarray, shape(ns,nt)
        Cost_matrix
    G : ndarray, shape (na,nb)
        OT matrix
    serie : string
        directory folder.

    Returns
    -------
    None.

    """
    ns=Xs.shape[0]
    nt=Xt.shape[0]
    
    # Plot 
    cols,colt=tools.label_position(Xs,Xt)
    display.plot_dots2(Xs,Xt,xs_color=cols,xt_color=colt)
    tools.save_fig('Initial_space',directory=serie)
    
    #Affichage cout de la transformation
    plt.figure(figsize=(20, 20))
    plt.imshow(M, interpolation='nearest')
    plt.title('Cost matrix M')
    tools.save_fig('Cost_matrix',directory=serie)
    
    # Transformation
    plt.figure(figsize=(20, 20))
    plt.imshow(G, interpolation='nearest')
    plt.title('OT_transport')
    plt.xlim(0,nt) 
    plt.ylim(0,ns)
    tools.save_fig('OT_transport',directory=serie)
    
    # Affichage détaillé
    display.plot_dots_links(Xs,Xt,G,degree_link=1,degree_label=1,color_weight=0.2,xs_color=cols)
    tools.save_fig('links',directory=serie)
    display.plot_dots_projection(Xs,Xt,G,xs_color=cols,xt_color=colt)
    tools.save_fig('labelling',directory=serie)
    
    print('auto_affichage_discret done '+str(serie))
    
def auto_affichage_continue(Xs,a,Xt,b,M,G,serie):
    """ Automatisation display processus for map

    Parameters
    ----------
    Xs : ndarray, shape (ns,2)
        Source samples positions
    a : numpy.ndarray, shape (ns,1)
        Liste des amplitudes associées aux coordonnées normalisées sous la forme de (y,x)   
    Xt : ndarray, shape (nt,2)
        Target samples positions
    b : numpy.ndarray, shape (nt,1)
        Liste des amplitudes associées aux coordonnées normalisées sous la forme de (y,x)   
    M : ndarray, shape(ns,nt)
        Cost_matrix
    G : ndarray, shape (na,nb)
        OT matrix
    serie : string
        directory folder.

    Returns
    -------
    None.

    """
    ns=Xs.shape[0]
    nt=Xt.shape[0]
    
    # Plot 
    cols,colt=tools.label_amplitude(a,b)
    display.plot_dots2(Xs,Xt,xs_color=cols,xt_color=colt)
    tools.save_fig('Initial_space',directory=serie)
    
    #Affichage cout de la transformation
    plt.figure(figsize=(20, 20))
    plt.imshow(M, interpolation='nearest')
    plt.title('Cost matrix M')
    tools.save_fig('Cost_matrix',directory=serie)
    
    # Transformation
    plt.figure(figsize=(20, 20))
    plt.imshow(G, interpolation='nearest')
    plt.title('OT_transport')
    plt.xlim(0,nt) 
    plt.ylim(0,ns)
    tools.save_fig('OT_transport',directory=serie)
    
    # Affichage détaillé
    display.plot_dots_links(Xs,Xt,G,degree_link=1,degree_label=1,color_weight=0.2,xs_color=cols)
    tools.save_fig('links',directory=serie)
    display.plot_dots_projection(Xs,Xt,G,xs_color=cols,xt_color=colt)
    tools.save_fig('labelling',directory=serie)
    print('auto_affichage_continue done '+str(serie))
    
def auto_processing(Xs,a,Xt,b,G,directory,affixe):  
    """ Automatisation save variables processus
    

    Parameters
    ----------
    Xs : ndarray, shape (ns,2)
        Source samples positions
    a : numpy.ndarray, shape (ns,1)
        Liste des amplitudes associées aux coordonnées normalisées sous la forme de (y,x)   
    Xt : ndarray, shape (nt,2)
        Target samples positions
    b : numpy.ndarray, shape (nt,1)
        Liste des amplitudes associées aux coordonnées normalisées sous la forme de (y,x)   
    G : ndarray, shape (na,nb)
        OT matrix
    directory : string
        directory folder.
    affixe : string
        name of variables

    Returns
    -------
    None.

    """
    if directory[-1]!='/':
        directory=directory+'/'
    tools.save_value(G,str(affixe),directory)
    #process.auto1(Xs,Xt,M,G0,'Experience 1/0')
    pos1_G,w1_G=tools.degree(Xs,Xt,G,degree=1)
    tools.save_value(pos1_G,'pos1_'+str(affixe),directory+str(affixe))
    tools.save_value(w1_G,'w1_'+str(affixe),directory+str(affixe))
    
    pos2_G,w2_G=tools.degree(Xs,Xt,G,degree=2)
    tools.save_value(pos2_G,'pos2_'+str(affixe),directory+str(affixe))
    tools.save_value(w2_G,'w2_'+str(affixe),directory+str(affixe))
    print('Auto_processing done '+str(directory)+str(affixe))
