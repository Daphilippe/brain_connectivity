# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 30/03/2021
@version: 1.0
@Recommandation: Python 3.7
@But: Discret
"""
import tools
import process
import ot
import sys
"""Discret"""
S,T=tools.data_generator()
Xs,a=S
Xt,b=T

# Processing
# Cost matrix
M = ot.dist(Xs, Xt)
M /= M.max()
tools.save_value(M,'M',directory='Discret 1')

G0 = ot.emd(a, b, M) #résolver sans terme de régularisation, on a bien choisit la fonction de cout associé à la norme 2
racine='discret_1'
process.auto_processing(Xs,a,Xt,b,G0,racine,'G0')  
process.auto_affichage_discret(Xs,Xt,M,G0,str(racine)+'/G0/images')

for lambd in [0.001,0.01,0.1,1,10,100]:
    temp = ot.sinkhorn(a, b, M, lambd)
    process.auto_processing(Xs,a,Xt,b,temp,str(racine),'G'+str(lambd))  
    process.auto_affichage_discret(Xs,Xt,M,temp,str(racine)+'/G'+str(lambd)+'/images')
sys.exit()