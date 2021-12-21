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


def PRIVIX(S,d,hashes,signs):     # estimate full vector
    t,k = S.shape
    x_tilde=np.zeros(d)

    for i in range(d):
        V=[signs[j,i]*S[j,hashes[j,i]] for j in range(t)]
        x_tilde[i] = np.median(V)

    return x_tilde


def HEAVYMIX(S, frac_topk, d,hashes,signs,round2,true_vec):
    """ Return the largest k elements (by magnitude) of Sketch"""
    t,k = S.shape
    x_tilde=np.zeros(d)
    S_unsketch = np.zeros(d)
    
    topk = round(frac_topk*d)
    for i in range(d):
        V=[signs[j,i]*S[j,hashes[j,i]] for j in range(t)]
        S_unsketch[i] = np.median(V)

    topkIndices = torch.sort(torch.from_numpy(S_unsketch)**2)[1][-topk:,]
    
    if round2:
        x_tilde[topkIndices]=true_vec[topkIndices]
    else:
        x_tilde[topkIndices] = S_unsketch[topkIndices]

    return x_tilde

def compress(vec):
        """
        :param vec: torch tensor
        :return: norm, signs, quantized_intervals
        """
        vec = vec.view(-1, self.dim)
        # norm = torch.norm(vec, dim=1, keepdim=True)
        norm = torch.max(torch.abs(vec), dim=1, keepdim=True)[0]
        normalized_vec = vec / norm

        scaled_vec = torch.abs(normalized_vec) * self.s
        l = torch.clamp(scaled_vec, 0, self.s-1).type(self.code_dtype)

        if self.random:
            # l[i] <- l[i] + 1 with probability |v_i| / ||v|| * s - l
            probabilities = scaled_vec - l.type(torch.float32)
            r = torch.rand(l.size())
            if self.cuda:
                r = r.cuda()
            l[:] += (probabilities > r).type(self.code_dtype)

        signs = torch.sign(vec) > 0
        return [norm, signs.view(self.shape), l.view(self.shape)]