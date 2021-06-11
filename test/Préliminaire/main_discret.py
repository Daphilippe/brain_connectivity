# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 08/04/2021
@version: 2.00
@Recommandation: Python 3.7
@revision: 11/06/2021
@But: Présentation module
"""
import numpy as np
import ot

import sys
sys.path.insert(1,'../../libs')
import tools, display, process

# Directory
racine='Test/'

# Create data xs and xt (dot map)
(xs,a),(xt,b)=tools.data_generator_simulation1()    
# Labelling by position 
cols,colt=tools.label_position(xs,xt)

# Display data
display.plot_dots2(xs,xt,xs_color=cols,xt_color=colt,size=(20,20))

# Compute cost matrix with sqeuclidean distance    
M = ot.dist(xs,xt)
# Compute OT matrix with emd algorithm
G0 = ot.emd(a, b, M, numItermax=1000000)

if False:
    affixe='G0'
    
    pos1,w1=tools.degree(xs,xt,G0,degree=1)
    
    # Save label
    tools.save_value(pos1,'pos1_'+str(affixe),racine+str(affixe))
    tools.save_value(w1,'w1_'+str(affixe),racine+str(affixe))
    
    # Recatch label
    pos1_G0=np.load(racine+affixe+'/pos1_'+affixe+'.npy')
    w1_G0=np.load(racine+affixe+'/w1_'+affixe+'.npy')
    
    # Reverse label
    # pos1_G0bis=pos1_G0.copy() #reverse sans recalcule du transport inverse
    # pos1_G0bis[:, 0], pos1_G0bis[:, 1] = pos1_G0[:, 1], pos1_G0[:, 0]
    
    # Save figure
    serie=racine+affixe+'/Figures'
    process.auto_affichage_discret(xs,xt,M,G0,pos1_G0,w1_G0,serie=serie)
    process.auto_affichage_continue(xs,a,xt,b,M,G0,pos=pos1_G0,weight=w1_G0,serie=serie,label='both')
else:
    affixe='G0'
    
    process.automatisation_discret(xs,a,xt,b,M,G0,racine,affixe)

sys.exit()