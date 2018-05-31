import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import time
import pdb
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


####################################
    ## Dict. Learning ##
####################################

# Class is the dictionary that you learn, is equipped with a a forward pass
class DictLearnt_IHT(nn.Module):
    def __init__(self, m, K):
        super(DictLearnt_IHT, self).__init__()
        self.W = nn.Parameter(torch.randn(28*28, m, requires_grad=False))        
        # normalization
        self.W.data = NormDict(self.W.data)
        self.K = K
        self.forward_type = "IHT"
        self.m = m
        self.mask = 1
        
    def forward(self, Y, K):       
        # normalizing Dict
        self.W.requires_grad_(False)
        self.W.data = NormDict(self.W.data)       
        # Sparse Coding
        Gamma, residual, errIHT = IHT(Y,self.W, self.mask, K)       
        # Reconstructing
        self.W.requires_grad_(True)
        X = torch.mm(Gamma,self.W.transpose(1,0))       
        # Sparsity
        NNZ = np.count_nonzero(Gamma.cpu().data.numpy())/Gamma.shape[0]
        return X, Gamma, errIHT
        

        
#--------------------------------------------------------------
#         Supporting Functions
#--------------------------------------------------------------

def hard_threshold_k(X, k):
    Gamma = X.clone()
    m = X.data.shape[1]
    a,_ = torch.abs(Gamma).data.sort(dim=1,descending=True)
    T = torch.mm(a[:,k].unsqueeze(1),torch.Tensor(np.ones((1,m))).to(device))
    mask = Variable(torch.Tensor((np.abs(Gamma.data.cpu().numpy())>T.cpu().numpy()) + 0.)).to(device)
    Gamma = Gamma * mask
    return Gamma#, mask.data.nonzero()


#--------------------------------------------------------------


def IHT(Y,W,mask,K):
    c = PowerMethod(W)
    # print(c)
    eta = 2/c
    ht_arg = dropout(torch.mm(Y,eta*W), mask)
    Gamma = hard_threshold_k(ht_arg, K)
    # plt.spy(Gamma); plt.show()
    # pdb.set_trace()   
    residual = torch.mm(Gamma, W.transpose(1,0)) - Y
    IHT_ITER = 50  
    norms = np.zeros((IHT_ITER,))
    for i in range(IHT_ITER):
        ht_arg = dropout((Gamma - eta * torch.mm(residual, W)), mask)
        Gamma = hard_threshold_k(ht_arg, K)
        residual = torch.mm(Gamma, W.transpose(1,0)) - Y
        norms[i] = np.linalg.norm(residual.cpu().numpy(),'fro')/ np.linalg.norm(Y.cpu().numpy(),'fro')   
    return Gamma, residual, norms


# def IHT(Y,W,K):
    
#     c = PowerMethod(W)
#     # print(c)
#     eta = 2/c
#     Gamma = hard_threshold_k(torch.mm(Y,eta*W),K)
#     # plt.spy(Gamma); plt.show()
#     # pdb.set_trace()
    
#     residual = torch.mm(Gamma, W.transpose(1,0)) - Y
#     IHT_ITER = 50
    
#     norms = np.zeros((IHT_ITER,))

#     for i in range(IHT_ITER):
#         Gamma = hard_threshold_k(Gamma - eta * torch.mm(residual, W), K)
#         residual = torch.mm(Gamma, W.transpose(1,0)) - Y
#         norms[i] = np.linalg.norm(residual.cpu().numpy(),'fro')/ np.linalg.norm(Y.cpu().numpy(),'fro')
    
#     return Gamma, residual, norms

#--------------------------------------------------------------

def NormDict(W):
    Wn = torch.norm(W, p=2, dim=0).detach()
    W = W.div(Wn.expand_as(W))
    return W

#--------------------------------------------------------------

def PowerMethod(W):
    ITER = 100
    m = W.shape[1]
    X = torch.randn(1, m).to(device)
    for i in range(ITER):
        Dgamma = torch.mm(X,W.transpose(1,0))
        X = torch.mm(Dgamma,W)
        nm = torch.norm(X,p=2)
        X = X/nm   
    return nm

#--------------------------------------------------------------

def sample_filters(numb_atoms, p, k):
    numb_active_filters = int(np.maximum(np.ceil(p*numb_atoms), k))
    active_filter_inds = random.sample(range(0, numb_atoms), numb_active_filters)
    return active_filter_inds

#--------------------------------------------------------------

def create_dropout_mask(batch_size, m, active_filter_inds):
    mask = torch.zeros(batch_size, m)
    for i in active_filter_inds:
        mask[:,i] = torch.ones(batch_size)
    return mask

#--------------------------------------------------------------

def dropout(X, mask):
    X_dropout = X*mask
    return X_dropout

#--------------------------------------------------------------

# this function might be buggy - needs checking
# def Threshold_Tensor(Tensor_in,K):
#     Gamma = Tensor_in.clone().detach()
#     a,_ = torch.abs(Gamma).data.sort(dim=1,descending=True)
#     a = a.to(device)
#     # if cudaopt:
#     #     T = torch.mm(a[:,K].unsqueeze(1),torch.Tensor(np.ones((1,m))).cuda())
#         # mask = Variable(torch.Tensor( ( torch.abs(Gamma).cpu().data.numpy()>T.cpu().numpy() ) + 0.).cuda())
#     # else:
#     m = Tensor_in.shape[1]
#     T = torch.mm(a[:,K].unsqueeze(1),torch.Tensor(np.ones((1,m))).to(device)).detach()
#     mask = torch.Tensor( (np.abs(Gamma.cpu().numpy())>T.cpu().numpy()) + 0.)
#     return (Tensor_in.clone() * mask.to(device))


