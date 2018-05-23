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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#--------------------------------------------------------------

def hard_threshold_k(X, k):
    Gamma = X.clone()
    m = X.data.shape[1]
    a,_ = torch.abs(Gamma).data.sort(dim=1,descending=True)
    T = torch.mm(a[:,k].unsqueeze(1),torch.Tensor(np.ones((1,m))).to(device))
    mask = Variable(torch.Tensor((np.abs(Gamma.data.cpu().numpy())>T.cpu().numpy()) + 0.)).to(device)
    Gamma = Gamma * mask
    return Gamma#, mask.data.nonzero()


# this function might be buggy - needs checking
def Threshold_Tensor(Tensor_in,K):
    Gamma = Tensor_in.clone().detach()
    a,_ = torch.abs(Gamma).data.sort(dim=1,descending=True)
    a = a.to(device)
    # if cudaopt:
    #     T = torch.mm(a[:,K].unsqueeze(1),torch.Tensor(np.ones((1,m))).cuda())
        # mask = Variable(torch.Tensor( ( torch.abs(Gamma).cpu().data.numpy()>T.cpu().numpy() ) + 0.).cuda())
    # else:
    m = Tensor_in.shape[1]
    T = torch.mm(a[:,K].unsqueeze(1),torch.Tensor(np.ones((1,m))).to(device)).detach()
    mask = torch.Tensor( (np.abs(Gamma.cpu().numpy())>T.cpu().numpy()) + 0.)
    return (Tensor_in.clone() * mask.to(device))


####################################
    ## K-Sparse AutoEncoder ##
####################################

class AutoEncoder(nn.Module):
    def __init__(self,m):
        super(AutoEncoder, self).__init__()

        self.W = nn.Parameter(torch.randn(28*28, m), requires_grad=True)
        self.b1 = nn.Parameter(torch.ones(1,m), requires_grad=True)
        self.b2 = nn.Parameter(torch.ones(1,28*28), requires_grad=True)
        
        # normalization
        self.W.data = 0.01 * self.W.data
        
    def forward(self, x, K):
        # Encoding
        
        encoded = torch.mm(x,self.W) + self.b1
            
        # Tresholding - local efficient
        res = Threshold_Tensor(encoded,K)
        
        # Reconstructing
        decoded = torch.mm(res,self.W.transpose(1,0))+self.b2
        
        # sparsity
        NNZ = np.count_nonzero(res.cpu().data.numpy())/res.shape[0]
        return encoded, decoded, NNZ, res
        