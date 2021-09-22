# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9  2020

@author: Xiaoyun Li
"""
import math
import numpy as np
import copy
import torch
import pdb
import collections

def generate_sketch(x,t,k,hashes,signs):    # suppose x is a numpy vector, e.g. reshaped local model
    d=len(x)
    
    S=np.zeros([t,k])
    
    for j in range(t):
        for i in range(d):
            S[j,hashes[j,i]]=S[j,hashes[j,i]]+signs[j,i]*x[i];

    return S


def aggregate_sketches(w_local,t,k,hashes,signs):     # w_local is num_worker by d numpy matrix containing models of multiple workers    
    n,d=w_local.shape
    S_global=np.zeros([t,k])
    
    for i in range(n):
        S_local=generate_sketch(w_local,t,k,hashes,signs)
        S_global=S_global+S_local/n
    
    return S_global


def PRIVIX(S,d,hashes,signs):     # estimate full vector
    t,k = S.shape
    x_tilde=np.zeros(d)

    for i in range(d):
        V=[signs[j,i]*S[j,hashes[j,i]] for j in range(t)]
        x_tilde[i] = np.median(V)

    return x_tilde

def HEAVYMIX(S, frac_topk, d,hashes,signs):
    """ Return the largest k elements (by magnitude) of Sketch"""
    t,k = S.shape
    x_tilde=np.zeros(d)
    S_unsketch = np.zeros(d)
    
    topk = round(frac_topk*d)
    for i in range(d):
        V=[signs[j,i]*S[j,hashes[j,i]] for j in range(t)]
        S_unsketch[i] = np.median(V)

    topkIndices = torch.sort(torch.from_numpy(S_unsketch)**2)[1][-topk:,]
    x_tilde[topkIndices] = S_unsketch[topkIndices]
    
    return x_tilde

# def HEAVYMIX(S, frac_topk, d,hashes,signs,round2,true_vec):
#     """ Return the largest k elements (by magnitude) of Sketch"""
#     t,k = S.shape
#     x_tilde=np.zeros(d)
#     S_unsketch = np.zeros(d)
    
#     topk = round(frac_topk*d)
#     for i in range(d):
#         V=[signs[j,i]*S[j,hashes[j,i]] for j in range(t)]
#         S_unsketch[i] = np.median(V)

#     topkIndices = torch.sort(torch.from_numpy(S_unsketch)**2)[1][-topk:,]
    
#     if round2:
#         x_tilde[topkIndices]=true_vec[topkIndices]
#     else:
#         x_tilde[topkIndices] = S_unsketch[topkIndices]

#     return x_tilde


def HEAPRIX(S,vec, frac_topk, d,hashes,signs):
    """ Return a combination of PRIVIX and HEAVYMIX of Sketch"""
    t,k = S.shape
    #first term in heaprix
    heav = HEAVYMIX(S,frac_topk, d,hashes,signs)

    #argument in second term term in heaprix (sketch of vector)
    arg_priv = vec - heav
    arg_priv_sketched = generate_sketch(arg_priv, t, k , hashes, signs)
    priv = PRIVIX(arg_priv_sketched,d,hashes,signs)

    #sum of two terms to return
    ret = heav + priv
    return ret

