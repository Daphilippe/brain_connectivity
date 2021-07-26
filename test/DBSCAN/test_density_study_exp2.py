# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 26/07/21
@version: 1.00
@Recommandation: Python 3.7
@But : Study of density
"""
import numpy as np
import sys
sys.path.insert(1,'../../libs')

import matplotlib.pylab as plt

import tools, display, barycenter, process

def triangle_sup(mask,n):
    # triangle supérieure de taille n (en fonction de l'axe x), masque d'une matrice carrée
    for i in range(0,n):#vertical
        for j in range(0,n):#horizontal
            if i+j<n:
                mask[i,j]=1
    return mask

def triangle_inf(mask,n):
    # triangle inférieur de taille n (en fonction de l'axe x), masque d'une matrice carrée
    for i in range(n,np.shape(mask)[0]):#vertical
        for j in range(n,np.shape(mask)[1]):#horizontal
            if (j+i-np.shape(mask)[0])>n:
                mask[i,j]=1
    return mask

# Directory
hemi='L'
source='../../data/'+hemi+'/'
source1='../../variables/'
chemin=source1+'DBSCAN/'

# Importation données
barycentre=np.load(source1+"barycentre/"+hemi+'/99.npy')
fenetre=31
Lglissant=np.load(source1+'isomap/'+hemi+'/barycentre glissant/bary_glissant_'+str(fenetre)+'_'+hemi+'.npy')

# Barycentre
X=barycentre
_,_,img_xs=tools.estimate_pseudo_density(X)
img_xs=img_xs/np.sum(img_xs)
image=img_xs#(img_xs>np.percentile(img_xs,90))*img_xs

if False:#Etude loi marginale pour détermination des 4 zones
    display.show_map(image, hemi+' - test')
    postcentral=np.sum(image,axis=1)
    precentral=np.sum(image,axis=0)
    
    plt.figure()
    #plt.scatter(list(range(0,len(precentral))),precentral,marker='*',label='Precentral')
    plt.scatter(list(range(0,len(postcentral))),postcentral,marker='+',label='Postcentral')
    plt.legend()
    #plt.title("Marginal laws")
    plt.ylim(0,np.max([precentral,postcentral])*1.1)
    plt.show()

mask1=triangle_sup(np.zeros(np.shape(image)) ,40)
mask4=triangle_inf(np.zeros(np.shape(image)) ,65)
mask2=triangle_sup(np.zeros(np.shape(image)) ,100)-mask1
mask3=triangle_inf(np.zeros(np.shape(image)) ,0)-mask4

if False: # affiche les masques
    display.show_map(mask1,'mask1')
    display.show_map(mask2,'mask2')
    display.show_map(mask3,'mask3')
    display.show_map(mask4,'mask4')

if True:#Etude densité
    Larea1=[]
    Larea2=[]
    Larea3=[]
    Larea4=[]
    for X in Lglissant:
        _,_,img_xs=tools.estimate_pseudo_density(X)
        img_xs=img_xs/np.sum(img_xs)
        
        Larea1.append(np.sum(img_xs*mask1))
        Larea2.append(np.sum(img_xs*mask2))
        Larea3.append(np.sum(img_xs*mask3))
        Larea4.append(np.sum(img_xs*mask4))

if True:# Affichage des figures de suivies
    plt.figure()
    plt.scatter(list(range(0,len(Larea1))),Larea1,label='area 1 - ventral',marker='+')
    plt.scatter(list(range(0,len(Larea1))),Larea2,label='area 2 - ventral',marker='+')
    plt.scatter(list(range(0,len(Larea1))),Larea3,label='area 3 - dorsal',marker='*')
    plt.scatter(list(range(0,len(Larea1))),Larea4,label='area 4 - dorsal',marker='*')
    plt.legend()
    plt.title(hemi+" - Density tracking")
    plt.ylim(0,np.max([Larea1,Larea2,Larea3,Larea4])*1.1)
    plt.show()