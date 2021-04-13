# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 30/03/2021
@version: 1.4
@revision: 09/04/2021
@Recommandation: Python 3.7
"""
import display
import tools

import numpy as np
import matplotlib.pylab as plt

def auto_affichage_discret(Xs,Xt,M,G,pos,weight=None,serie=''):
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
        directory folder. The default is ''.

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
    
    display.plot_dots_links(Xs,Xt,pos,weight,xs_color=cols)
    tools.save_fig('links',directory=serie)
    display.plot_dots_projection(Xs,Xt,pos,xs_color=cols,xt_color=colt)
    tools.save_fig('labelling',directory=serie)
    
    print('auto_affichage_discret done '+str(serie))
    
def auto_affichage_continue(Xs,a,Xt,b,M,G,pos,weight=None,serie='',label='both'):
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
    pos : ndarray, shape (nt,2)
        Correspondance xt dots with ns dots. We don't nessesary have bijection between ns and nt
    weight : ndarray, shape (nt,2)
        weight of bridge between xt dots with ns dots. We don't nessesary have bijection between ns and nt
    serie : string
        directory folder.
    label : string
        Choice : 'amplitude', 'position', 'both'
        The default is both.

    Returns
    -------
    None.

    """
    ns=Xs.shape[0]
    nt=Xt.shape[0]
    
    # Plot      
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
    
    if label=='position' or label=='both':
        ## Position
        cols,colt=tools.label_position(Xs,Xt)
        display.plot_dots2(Xs,Xt,xs_color=cols,xt_color=colt)
        tools.save_fig('Initial_space_pos',directory=serie)
        
        # Affichage détaillé
        display.plot_dots_links(Xs,Xt,pos,weight,xs_color=cols)
        tools.save_fig('links_pos',directory=serie)
        display.plot_dots_projection(Xs,Xt,pos,xs_color=cols,xt_color=colt)
        tools.save_fig('labelling_pos',directory=serie)
    
    if label=='amplitude' or label=='both':
        ## Amplitude
        cols,colt=tools.label_amplitude(a,b)
        display.plot_dots2(Xs,Xt,xs_color=cols,xt_color=colt)
        tools.save_fig('Initial_space_ampl',directory=serie)
        
        # Affichage détaillé
        display.plot_dots_links(Xs,Xt,pos,weight,xs_color=cols)
        tools.save_fig('links_ampl',directory=serie)
        display.plot_dots_projection(Xs,Xt,pos,xs_color=cols,xt_color=colt)
        tools.save_fig('labelling_ampl',directory=serie)
    print('auto_affichage_continue done '+str(serie))
    
def auto_processing(Xs,a,Xt,b,G,directory='',affixe=''):  
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
        directory folder. The default is ''.
    affixe : string
        name of variables. The default is ''.

    Returns
    -------
    None.

    """
    if directory[-1]!='/':
        directory=directory+'/'
    tools.save_value(G,str(affixe),directory)
    pos1_G,w1_G=tools.degree(Xs,Xt,G,degree=1)
    tools.save_value(pos1_G,'pos1_'+str(affixe),directory+str(affixe))
    tools.save_value(w1_G,'w1_'+str(affixe),directory+str(affixe))
    
    pos2_G,w2_G=tools.degree(Xs,Xt,G,degree=2)
    tools.save_value(pos2_G,'pos2_'+str(affixe),directory+str(affixe))
    tools.save_value(w2_G,'w2_'+str(affixe),directory+str(affixe))
    print('Auto_processing done '+str(directory)+str(affixe))

def automatisation_continue(Xs,a,Xt,b,M,G,racine,affixe):
    auto_processing(Xs,a,Xt,b,G,racine,affixe)  
    pos1=np.load(str(racine)+'/'+str(affixe)+'/pos1_'+str(affixe)+'.npy')
    w1=np.load(str(racine)+'/'+str(affixe)+'/w1_'+str(affixe)+'.npy')
    auto_affichage_continue(Xs,a,Xt,b,M,G,pos1,w1,str(racine)+'/G0/images',label='both')
    
def automatisation_discret(Xs,a,Xt,b,M,G,racine,affixe):
    auto_processing(Xs,a,Xt,b,G,racine,affixe)  
    pos1=np.load(str(racine)+'/'+str(affixe)+'/pos1_'+str(affixe)+'.npy')
    w1=np.load(str(racine)+'/'+str(affixe)+'/w1_'+str(affixe)+'.npy')
    auto_affichage_discret(Xs,Xt,M,G,pos1,w1,str(racine)+'/'+str(affixe)+'/images') 