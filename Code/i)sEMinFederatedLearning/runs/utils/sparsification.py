# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 00:45:37 2019

@author: bdfzl
"""

import numpy as np
import torch

def compute_prob(g,u,density,order):
    size = g.shape
    g = torch.flatten(g).numpy()
    u = torch.flatten(u).numpy()
    d=len(g)
    c = np.round(d*density)
    p_out = np.ones([d])
    active_ind = np.array(list(range(d)))
    active_set = np.array(list(range(d)))
    
    rounds=0
    while True:
#        rounds=rounds+1
#        if rounds>3:
#            break
        
        ratio = np.abs(g[active_set])/(u[active_set]**order)
        p = c * ratio / np.sum(ratio)
        if np.max(np.abs(g[active_set]))==0:
            return np.zeros([d])
        
        if np.max(p)<=1:
            break
        
        active_ind = np.where(p<1)[0]
        k= len(active_set)-len(active_ind)
        c = c - k
        active_set = active_set[active_ind]
    
    p_out[active_set] = p
    p_out = np.reshape(p_out,size)
    
    return p_out

def compute_g_tilde(g,p):
    gate = np.random.binomial(1,p)
    p[p==0]=1
    g_tilde = g / p * gate
    
    return g_tilde

def compute_g_tilde_unif(g,density):
    p=density*np.ones_like(g)
    gate = np.random.binomial(1,p)
    g_tilde = g / p * gate
    
    return g_tilde
        
def compute_prob_Tong(g,density):
    size = g.shape
    g = torch.flatten(g).numpy()
    d=len(g)
    c = np.round(d*density)
    p_out = np.ones([d])
    active_ind = np.array(list(range(d)))
    active_set = np.array(list(range(d)))
    
    while True:
        ratio = np.abs(g[active_set])
        p = c * ratio / np.sum(ratio)
        if np.max(np.abs(g[active_set]))==0:
            return np.zeros([d])
        
        if np.max(p)<=1:
            break
        
        active_ind = np.where(p<1)[0]
        k= len(active_set)-len(active_ind)
        c = c - k
        active_set = active_set[active_ind]
    
    p_out[active_set] = p
    p_out = np.reshape(p_out,size)
    
    return p_out  

def compute_topk(g,density):
    g=g.numpy()
    
    threshold = np.quantile(np.abs(g),1-density)
    g[np.abs(g)<=threshold]=0
    g=torch.from_numpy(g)
    
    return g
    
def sign_grad(g):
    g=g.numpy()
    
    scale = np.mean(np.abs(g))
    return torch.from_numpy(np.sign(g)*scale)

#def body(c,d,g,u,active_set,active_ind,p,p_out):
#    this_g = tf.gather(g,active_set)
#    this_u = tf.gather(u,active_set)
#    ratio = tf.abs(this_g)/(this_u**(1/4))
#    p = c * ratio / tf.reduce_sum(ratio)
##        if tf.equal(tf.reduce_max(tf.abs(this_g)),0):
##            p_out = tf.zeros([d])
#    
#    active_ind = tf.where(tf.less(p,1))
#    aa = active_set.shape;bb = active_ind.shape
#    print('c: ',c)
#    print('aa: ',aa[0])
#    print('bb: ',bb)
#    k= aa[0].value-bb[0].value
#    c =  c - k
#    active_set = tf.gather_nd(active_set,active_ind)
#    
#    return p,p_out,active_set
#
#def condition(c,d,g,u,active_set,active_ind,p,p_out):
#    return tf.less_equal(tf.reduce_max(p),1)
#
#def tf_generate_tilde_g(g,u,c):
#    
#    tensor_size = g.get_shape()
#    d = tf.reduce_prod(tensor_size)
#    g_reshaped = tf.reshape(g,[-1])
#    u_reshaped = tf.reshape(u,[-1])
#    
#    p_out = tf.Variable(tf.ones([d]),trainable=False)
#    
#    active_ind = tf.range(0,d,1)
#    active_set = tf.range(0,d,1)
#    
#    p = tf.ones([d])
#    
#    this_g = tf.gather(g_reshaped,active_set)
#    p_out = tf.cond(tf.equal(tf.reduce_max(tf.abs(this_g)),0),lambda: tf.zeros([d]),lambda: p_out)  
#    
##    while True:
##        this_g = tf.gather_nd(g_reshaped,active_set)
##        this_u = tf.gather_nd(u_reshaped,active_set)
##        ratio = tf.abs(this_g)/(this_u**(1/4))
##        p = c * ratio / tf.reduce_sum(ratio)
###        if tf.equal(tf.reduce_max(tf.abs(this_g)),0):
###            p_out = tf.zeros([d])
##        p_out = tf.cond(tf.equal(tf.reduce_max(tf.abs(this_g)),0),tf.zeros([d]),p_out)
##        
##        if tf.less_equal(tf.reduce_max(p),1):
##            break
##    
##        active_ind = tf.where(tf.less(p,1))
##        k= active_set.shape[0]-active_ind.shape[0]
##        c = c - k
##        active_set = tf.gather_nd(active_set,active_ind)
#    
#    p,p_out,active_set = tf.while_loop(condition,body,[c,d,g_reshaped,u_reshaped,active_set,active_ind,p,p_out])
#    
#    p_out[active_set] = p
#    p_out = tf.tensor_scatter_update(p_out, active_set, p)
#    
#    bernoullis = tf.nn.relu(tf.sign(p_out - tf.random_uniform(tf.shape(p_out))))
#    
#    zero_ind = tf.where(tf.less(p_out,0))
#    p_out = tf.tensor_scatter_update(p_out, zero_ind, 0.01)
#    
#    g_tilde = g / p_out * bernoullis
#    
#    g_tilde = tf.reshape(g_tilde,tensor_size)
#    
#    return g_tilde