# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 08/04/2021
@version: 1.0
@Recommandation: Python 3.7
@But: Continue
"""
import tools
import process
import ot
import sys
"""Continue"""
S,T=tools.data_generator()
Xs,a=S
Xt,b=T

#Génération de la carte
_,_,Img_Xs=tools.estimate_pseudo_density(Xs)
_,_,Img_Xt=tools.estimate_pseudo_density(Xt)

# Continue vers discret
Img_Xs,Img_Xt=tools.continue2discret(Img_Xs,Img_Xt,seuil=0)
Img2Xs,aa=tools.extract_point(Img_Xs)
Img2Xt,bb=tools.extract_point(Img_Xt)

# Processing
# Cost matrix
M = ot.dist(Img2Xs, Img2Xt)
racine='continue_1'
tools.save_value(M,'M',directory=racine)

G0 = ot.emd(aa, bb, M, numItermax=1000000)
process.auto_processing(Xs,a,Xt,b,G0,racine,'G0')  
process.auto_affichage_continue(Img2Xs,aa,Img2Xt,bb,M,G0,str(racine)+'/G0/images')

for lambd in [0.001,0.01,0.1,1,10,100]:
    temp = ot.sinkhorn(aa,bb, M, lambd, numItermax=1000000)
    process.auto_processing(Xs,a,Xt,b,temp,racine,'G'+str(lambd))  
    process.auto_affichage_continue(Img2Xs,aa,Img2Xt,bb,M,temp,str(racine)+'/G'+str(lambd)+'/images')
sys.exit()