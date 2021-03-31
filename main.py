# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 30/03/2021
@version: 1.0
@Recommandation: Python 3.7
"""
import tools
import process
import ot

"""Discret"""
S,T=tools.data_generator()
Xs,a=S
Xt,b=T

# Processing
# Cost matrix
M = ot.dist(Xs, Xt)
M /= M.max()
tools.save_value(M,'M',directory='Experience 1')

G0 = ot.emd(a, b, M) #résolver sans terme de régularisation, on a bien choisit la fonction de cout associé à la norme 2
process.auto_processing(Xs,a,Xt,b,G0,'Experience 1','G0')  
process.auto_affichage_discret(Xs,Xt,M,G0,'Experience 1/G0/images')

lambd=0.01
G001 = ot.sinkhorn(a, b, M, lambd)
process.auto_processing(Xs,a,Xt,b,G001,'Experience 1','G001')  
process.auto_affichage_discret(Xs,Xt,M,G001,'Experience 1/G001/images')

"""Continue"""
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
tools.save_value(M,'M',directory='Experience 2')

G0 = ot.emd(aa, bb, M, numItermax=1000000)
process.auto_processing(Xs,a,Xt,b,G0,'Experience 2','G0')  
process.auto_affichage_continue(Img2Xs,aa,Img2Xt,bb,M,G0,'Experience 2/G0/images')

lambd=10
G10 = ot.sinkhorn(aa,bb, M, lambd, numItermax=1000000)
process.auto_processing(Xs,a,Xt,b,G10,'Experience 2','G10')  
process.auto_affichage_continue(Img2Xs,aa,Img2Xt,bb,M,G10,'Experience 2/G10/images')