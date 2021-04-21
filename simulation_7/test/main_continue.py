# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 08/04/2021
@version: 1.75
@Recommandation: Python 3.7
@revision: 21/04/2021
@But: Régularisation
"""
# -*- coding: utf-8 -*-
import numpy as np
import ot
import sys

sys.path.insert(1,'../libs')
import tools, display, process
    
# Directory
racine='Test/'

# Create data
source_dot=[]
source_map=[]
for i in range(2):
    # Génération des données
    temp=np.random.randint(2,5)
    S=tools.data_generator_simulation2(rand_seed=temp,noise=0.5*np.log(temp))
    xs,a=S
    source_dot.append([xs,a])
    
    # Génération de la carte
    _,_,Img_xs=tools.estimate_pseudo_density(xs)
    source_map.append(Img_xs)

# Save data
for i in range(len(source_map)):
    display.show_map(source_map[i],title='Map_'+str(i))
    tools.save_fig('source_'+str(i),directory=racine)

if True:
    ## Test régularisation
    # Chargement données et sauvegarde
    xs,a=source_dot[0]
    tools.save_value(xs,'xs',directory=racine)
    tools.save_value(a,'a',directory=racine)
    Img_xs=source_map[0]
    
    xt,b=source_dot[1]
    tools.save_value(xt,'xt',directory=racine)
    tools.save_value(b,'b',directory=racine)
    Img_xt=source_map[1]
    
    # Seuillage pour enlever les points parasites générés par la gaussienne
    Img_xs,Img_xt=tools.continue2discret(Img_xs,Img_xt,seuil=10e-4)
    # Extraction des points
    Img2xs,aa=tools.extract_point(Img_xs)
    Img2xt,bb=tools.extract_point(Img_xt)
    
    # Cost matrix
    M = ot.dist(Img2xs, Img2xt)
    tools.save_value(M,'M',directory=racine)

    # Sans régularisation
    print('Sans regularisation')
    G0 = ot.emd(aa, bb, M, numItermax=1000000)
    
    affixe='G0'    
    process.automatisation_continue(Img2xs,aa,Img2xt,bb,M,G0,racine,affixe)
else :
    # Test avec régularisation entropique + régularisation norme de Frobenius
    def f(G):
        #Frobenius regularization
        return 0.5 * np.sum(G**2)

    def df(G):
        return G
    # Double 
    print('Regularisation entropique')
    reg_lasso=10
    reg_entropique=10
    temp2=ot.optim.gcg(aa, bb, M, reg_entropique, reg_lasso, f, df)
    
    affixe= '2reg_'+str(reg_entropique)+'_'+str(reg_lasso)
    process.automatisation_continue(Img2xs,aa,Img2xt,bb,M,temp2,racine,affixe)
      
sys.exit()

