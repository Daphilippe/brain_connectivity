# -*- coding: utf-8 -*-
"""
@author: Duy Anh Philippe Pham
@date: 15/04/2021
@version: 1.75
@Recommandation: Python 3.7
@revision: 21/04/2021
@But: Centroide
@list of functions
- compute(source='./data/R/',directory='variables/')
- display(data_source='./data/R/',centroide_source='./variables/R/',directory='variables/')
- matrix(centroide_source='./variables/R/')
"""
import numpy as np
import ot

#import time
import glob
import sys

sys.path.insert(1,'../libs')
import tools

def compute(source='./data/R/',directory='variables/'):
    if directory[-1]!='/':
        directory=directory+'/'
        
    if source[-1]!='/':
        source=source+'/'
    
    numpy_vars = {}
    for np_name in glob.glob(str(source)+'*.np[yz]'):
        numpy_vars[np_name] = np.load(np_name)
        
    size=len(source)
    Lval=[]
    Lname=[]
    
    for i in numpy_vars:
        L2_val=[]
        L2_name=[] 
        
        for j in numpy_vars:
            if i==j:
                cost=0
                
            else:
                # M = ot.dist(xs,xt)
                # xs = numpy_vars[i]
                # xt = numpy_vars[j]
                M=ot.dist(numpy_vars[i],numpy_vars[j])
                
                # G = ot.emd(a,b,M, numItermax=1000000))
                # a = ot.emd(ot.unif(len(numpy_vars[i]))
                # b = ot.emd(ot.unif(len(numpy_vars[j]))

                # OT matrix
                G=ot.emd(ot.unif(len(numpy_vars[i])), ot.unif(len(numpy_vars[j])),M,numItermax=1000000)
                #G=ot.sinkhorn(ot.unif(len(numpy_vars[i])), ot.unif(len(numpy_vars[j])),M,reg=10,numItermax=30000)
                
                # Save OT matrix /i\ 
                #tools.save_value(G,'G_emd_'+str(j[9:17]),directory='variables/'+str(i[9:17]))
                
                # Wasserstein matrix
                cost=G*M  
                
            # update in between step
            L2_name.append(str(j[size:size+8]))
            # Wasserstein distance ^2
            L2_val.append(np.sum(cost))
            
        # Save in between step
        tools.save_value(L2_val,'L2_val_'+str(i[size:size+8]),directory+str(i[size:size+8]))
        tools.save_value(L2_name,'L2_name_'+str(i[size:size+8]),directory+str(i[size:size+8]))
        
        # update final step
        Lval.append(np.sum(L2_val))
        Lname.append(i)
        
    # Save final step
    tools.save_value(Lval,'L_val',directory)
    tools.save_value(Lname,'L_name',directory) 
    
    # Centroide
    argmin=np.argmin(Lval)
    centroide=Lname[argmin]
    tools.save_value(centroide,'centroide',directory)

def display(data_source='./data/R/',centroide_source='./variables/R/',directory='variables/'):
    import matplotlib.pylab as plt
    if data_source[-1]!='/':
        data_source=data_source+'/'
    if data_source[1]=='.':    
        size=len(data_source)-1
    
    if centroide_source[-1]!='/':
        centroide_source=centroide_source+'/'
        
    if directory[-1]!='/':
        directory=directory+'/'
        
    L_name=np.load(str(centroide_source)+'L_name.npy')
    L_val=np.load(str(centroide_source)+'L_val.npy')
    L_trie=[L_name[i][size:] for i in np.argsort(L_val)]
    
    fig=plt.figure(figsize=(100,100))
    for i in range(0,np.shape(L_trie)[0]):
        chemin=data_source+L_trie[i]
        
        a=fig.add_subplot(10,10,i+1)
        xs=np.load(chemin)
        a.title.set_text(str(L_trie[i][:8])+'-'+str(np.shape(xs)[0]))
        plt.xlim(0,100) 
        plt.ylim(0,100)
        plt.scatter(xs[:,0],xs[:,1],marker = '+', s = 50)    
    tools.save_fig('Centroide_dot',directory)
           
    fig=plt.figure(figsize=(100,100))
    for i in range(0,np.shape(L_trie)[0]):
        chemin=data_source+L_trie[i]
        a=fig.add_subplot(10,10,i+1)
        xs=np.load(chemin)
        _,_,Img_Xs=tools.estimate_pseudo_density(xs)
        a.title.set_text(str(L_trie[i][:8]))
        plt.xlim(0,100) 
        plt.ylim(0,100)
        plt.imshow(Img_Xs)
    tools.save_fig('Centroide_map',directory)
    
def matrix(centroide_source='./variables/R/'):
    import pandas as pd
    if centroide_source[-1]!='/':
        centroide_source=centroide_source+'/'
    size=len(centroide_source)
    numpy_vars = {}
    for i in list(zip(glob.glob(centroide_source+'*/L2_name*.np[yz]'),glob.glob(centroide_source+'*/L2_val*.np[yz]'))):
        name,val=i
        numpy_vars[name[size:size+8]] = np.load(val)
        df = pd.DataFrame(data=numpy_vars,index=np.load(name))
        
    df.sort_index()
    df = df.reindex(sorted(df.columns), axis=0)
    
    tools.save_value(df,'matrix',centroide_source)
    tools.save_value(df.index,'matrix_index',centroide_source)
    tools.save_value(df.columns,'matrix_columns',centroide_source)