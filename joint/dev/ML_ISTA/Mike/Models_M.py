import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import math


##################################################

####            Normal Neural Network           ####

##################################################

class ML_NN(nn.Module):
    def __init__(self,m1,m2,m3):
        super(ML_NN, self).__init__()
        
        # Convolutional Filters
        self.W1 = nn.Parameter(torch.randn(m1,1,6,6), requires_grad=True)
        self.strd1 = 2;
        self.W2 = nn.Parameter(torch.randn(m2,m1,6,6), requires_grad=True)
        self.strd2 = 2;
        self.W3 = nn.Parameter(torch.randn(m3,m2,4,4), requires_grad=True)
        self.strd3 = 1;
        
        # Biases / Thresholds
        self.b1 = nn.Parameter(torch.zeros(1,m1,1,1), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1,m2,1,1), requires_grad=True)
        self.b3 = nn.Parameter(torch.zeros(1,m3,1,1), requires_grad=True)
        
        # Classifier
        self.Wclass = nn.Linear(m3, 10)
        
        # Initialization
        self.W1.data = 0.01 * self.W1.data
        self.W2.data = 0.01 * self.W2.data
        self.W3.data = 0.01 * self.W3.data
        
    def forward(self, x):
        
        # Encoding
        gamma1 = F.relu(F.conv2d(x,self.W1, stride = self.strd1) + self.b1)       # first estimation
        gamma2 = F.relu(F.conv2d(gamma1,self.W2, stride = self.strd2) + self.b2) 
        gamma3 = F.relu(F.conv2d(gamma2,self.W3, stride = self.strd3) + self.b3) 
            
        # classifier
        gamma = gamma3.view(gamma3.shape[0],gamma3.shape[1]*gamma3.shape[2]*gamma3.shape[3])
        out = self.Wclass(gamma)
        out = F.log_softmax(out,dim = 1)
    
        return gamma, out, gamma1.data.cpu().numpy(), gamma2.data.cpu().numpy(), gamma3.data.cpu().numpy()
   


    #################################################################################################### 


##################################################

####        JNN with ReLU        ####

##################################################

class ML_JNN_ReLU(nn.Module):
    def __init__(self,m1,m2,m3):
        super(ML_JNN_ReLU, self).__init__()
        
        # Convolutional Filters
        self.W1 = nn.Parameter(torch.randn(m1,1,6,6), requires_grad=True)
        self.strd1 = 2;
        self.W2 = nn.Parameter(torch.randn(m2,m1,6,6), requires_grad=True)
        self.strd2 = 2;
        self.W3 = nn.Parameter(torch.randn(m3,m2,4,4), requires_grad=True)
        self.strd3 = 1;
        
        # Biases / Thresholds
        self.b1 = nn.Parameter(torch.zeros(1,m1,1,1), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1,m2,1,1), requires_grad=True)
        self.b3 = nn.Parameter(torch.zeros(1,m3,1,1), requires_grad=True)
        
        # Classifier
        self.Wclass = nn.Linear(m3, 10)
        
        # Initialization
        self.W1.data = 0.01 * self.W1.data
        self.W2.data = 0.01 * self.W2.data
        self.W3.data = 0.01 * self.W3.data
        

    
    def forward_joint(self, x, b_y):        
        # Encoding
        gamma1 = F.relu(F.conv2d(x,self.W1, stride = self.strd1) + self.b1)  
        gamma2 = F.relu(F.conv2d(gamma1,self.W2, stride = self.strd2) + self.b2) 
        X = F.conv2d(gamma2,self.W3, stride = self.strd3)
        gamma3 = torch.zeros(X.shape)
        if X.device.type=='cuda': gamma3 = gamma3.cuda()
        for i in range(10):
            n_ci = float(torch.sum(b_y==i))
            if n_ci>0:
                X_i = X[b_y==i,:,:,:]
                x_i = torch.sqrt(torch.sum(X_i**2,dim=0)) # computing norm of rows
                Z_i = F.relu(X_i*((n_ci) + self.b3/x_i))
                gamma3[b_y==i,:,:,:] = Z_i
        
        # classifier
        gamma = gamma3.view(gamma3.shape[0],gamma3.shape[1]*gamma3.shape[2]*gamma3.shape[3])
        out = self.Wclass(gamma)
        out = F.log_softmax(out,dim = 1)
    
        return gamma, out
         
    
    def forward(self, x):
        
        # Encoding
        gamma1 = F.relu(F.conv2d(x,self.W1, stride = self.strd1) + self.b1)      
        gamma2 = F.relu(F.conv2d(gamma1,self.W2, stride = self.strd2) + self.b2) 
        gamma3 = F.relu(F.conv2d(gamma2,self.W3, stride = self.strd3) + self.b3)
        
        # classifier
        gamma = gamma3.view(gamma3.shape[0],gamma3.shape[1]*gamma3.shape[2]*gamma3.shape[3])
        out = self.Wclass(gamma)
        out = F.log_softmax(out,dim = 1)
        
        
        return gamma, out, gamma1.data.cpu().numpy(), gamma2.data.cpu().numpy(), gamma3.data.cpu().numpy()
    
    
    
    
####################################################################################################   
    
    
##################################################

####            MultiLayer ISTA NET           ####

##################################################

class ML_ISTA(nn.Module):
    def __init__(self,m1,m2,m3):
        super(ML_ISTA, self).__init__()
        
        # Convolutional Filters
        self.W1 = nn.Parameter(torch.randn(m1,1,6,6), requires_grad=True)
        self.strd1 = 2;
        self.W2 = nn.Parameter(torch.randn(m2,m1,6,6), requires_grad=True)
        self.strd2 = 2;
        self.W3 = nn.Parameter(torch.randn(m3,m2,4,4), requires_grad=True)
        self.strd3 = 1;
        
        # Biases / Thresholds
        self.b1 = nn.Parameter(torch.zeros(1,m1,1,1), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1,m2,1,1), requires_grad=True)
        self.b3 = nn.Parameter(torch.zeros(1,m3,1,1), requires_grad=True)
        
        # Classifier
        self.Wclass = nn.Linear(m3, 10)
        
        # Initialization
        self.W1.data = 0.01 * self.W1.data
        self.W2.data = 0.01 * self.W2.data
        self.W3.data = 0.01 * self.W3.data
        
    def forward(self, x,T=0,RHO=1):
        
        # Encoding
        gamma1 = F.relu(F.conv2d(x,self.W1, stride = self.strd1) + self.b1)       # first estimation
        gamma2 = F.relu(F.conv2d(gamma1,self.W2, stride = self.strd2) + self.b2) 
        gamma3 = F.relu(F.conv2d(gamma2,self.W3, stride = self.strd3) + self.b3) 
        
        for _ in  range(T):
            
            # backward computatoin
            gamma2_ml = F.conv_transpose2d(gamma3,self.W3, stride=self.strd3)
            gamma1_ml = F.conv_transpose2d(gamma2_ml,self.W2, stride=self.strd2)
            
            gamma1 = (1-RHO) * gamma1 + RHO * gamma1_ml
            gamma2 = (1-RHO) * gamma2 + RHO * gamma2_ml
            
            # forward computation
            gamma1 = F.relu( (gamma1 - F.conv2d( F.conv_transpose2d(gamma1,self.W1, stride = self.strd1) - x ,self.W1, stride = self.strd1)) + self.b1)
            gamma2 = F.relu( (gamma2 - F.conv2d( F.conv_transpose2d(gamma2,self.W2, stride = self.strd2) - gamma1, self.W2, stride = self.strd2)) + self.b2) 
            gamma3 = F.relu( (gamma3 - F.conv2d( F.conv_transpose2d(gamma3,self.W3, stride = self.strd3) - gamma2, self.W3, stride = self.strd3)) + self.b3) 
            
        # classifier
        gamma = gamma3.view(gamma3.shape[0],gamma3.shape[1]*gamma3.shape[2]*gamma3.shape[3])
        out = self.Wclass(gamma)
        out = F.log_softmax(out,dim = 1)
    
        return gamma, out
   


    #################################################################################################### 


##################################################

####        MultiLayer J-ISTA with ReLU        ####

##################################################

class ML_JISTA_ReLU(nn.Module):
    def __init__(self,m1,m2,m3):
        super(ML_JISTA_ReLU, self).__init__()
        
        # Convolutional Filters
        self.W1 = nn.Parameter(torch.randn(m1,1,6,6), requires_grad=True)
        self.strd1 = 2;
        self.W2 = nn.Parameter(torch.randn(m2,m1,6,6), requires_grad=True)
        self.strd2 = 2;
        self.W3 = nn.Parameter(torch.randn(m3,m2,4,4), requires_grad=True)
        self.strd3 = 1;
        
        # Biases / Thresholds
        self.b1 = nn.Parameter(torch.zeros(1,m1,1,1), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1,m2,1,1), requires_grad=True)
        self.b3 = nn.Parameter(torch.zeros(1,m3,1,1), requires_grad=True)
        
        # Classifier
        self.Wclass = nn.Linear(m3, 10)
        
        # Initialization
        self.W1.data = 0.01 * self.W1.data
        self.W2.data = 0.01 * self.W2.data
        self.W3.data = 0.01 * self.W3.data
        

    
    def forward_joint(self, x, b_y,T=0,RHO=1):        
        # Encoding
        gamma1 = F.relu(F.conv2d(x,self.W1, stride = self.strd1) + self.b1)  
        gamma2 = F.relu(F.conv2d(gamma1,self.W2, stride = self.strd2) + self.b2) 
        X = F.conv2d(gamma2,self.W3, stride = self.strd3)
        gamma3 = torch.zeros(X.shape)
        if X.device.type=='cuda': gamma3 = gamma3.cuda()
        for i in range(10):
            n_ci = float(torch.sum(b_y==i))
            if n_ci>0:
                X_i = X[b_y==i,:,:,:]
                x_i = torch.sqrt(torch.sum(X_i**2,dim=0)) # computing norm of rows
                Z_i = F.relu(X_i*((n_ci) + self.b3/x_i))
                gamma3[b_y==i,:,:,:] = Z_i
     
        for _ in  range(T):
            
            # backward computatoin
            gamma2_ml = F.conv_transpose2d(gamma3,self.W3, stride=self.strd3)
            gamma1_ml = F.conv_transpose2d(gamma2_ml,self.W2, stride=self.strd2)
            
            gamma1 = (1-RHO) * gamma1 + RHO * gamma1_ml
            gamma2 = (1-RHO) * gamma2 + RHO * gamma2_ml
            
            # forward computation
            gamma1 = F.relu( (gamma1 - F.conv2d( F.conv_transpose2d(gamma1,self.W1, stride = self.strd1) - x ,self.W1, stride = self.strd1)) + self.b1)
            gamma2 = F.relu( (gamma2 - F.conv2d( F.conv_transpose2d(gamma2,self.W2, stride = self.strd2) - gamma1, self.W2, stride = self.strd2)) + self.b2) 
            
            X = (gamma3 - F.conv2d( F.conv_transpose2d(gamma3,self.W3, stride = self.strd3) - gamma2, self.W3, stride = self.strd3))
            gamma3 = gamma3*0
            for i in range(10):
                n_ci = float(torch.sum(b_y==i))
                if n_ci>0:
                    X_i = X[b_y==i,:,:,:]
                    x_i = torch.sqrt(torch.sum(X_i**2,dim=0)) # computing norm of rows 
                    Z_i = F.relu(X_i*((n_ci) + self.b3/x_i))
                    gamma3[b_y==i,:,:,:] = Z_i
        
        # classifier
        gamma = gamma3.view(gamma3.shape[0],gamma3.shape[1]*gamma3.shape[2]*gamma3.shape[3])
        out = self.Wclass(gamma)
        out = F.log_softmax(out,dim = 1)
    
        return gamma, out
    
     
    
    def forward(self, x,T=0,RHO=1):
        
        # Encoding
        gamma1 = F.relu(F.conv2d(x,self.W1, stride = self.strd1) + self.b1)      
        gamma2 = F.relu(F.conv2d(gamma1,self.W2, stride = self.strd2) + self.b2) 
        gamma3 = F.relu(F.conv2d(gamma2,self.W3, stride = self.strd3) + self.b3)
        
        for _ in  range(T):
            
            # backward computatoin
            gamma2_ml = F.conv_transpose2d(gamma3,self.W3, stride=self.strd3)
            gamma1_ml = F.conv_transpose2d(gamma2_ml,self.W2, stride=self.strd2)
            
            gamma1 = (1-RHO) * gamma1 + RHO * gamma1_ml
            gamma2 = (1-RHO) * gamma2 + RHO * gamma2_ml
            
            # forward computation
            gamma1 = F.relu( (gamma1 - F.conv2d( F.conv_transpose2d(gamma1,self.W1, stride = self.strd1) - x ,self.W1, stride = self.strd1)) + self.b1)
            gamma2 = F.relu( (gamma2 - F.conv2d( F.conv_transpose2d(gamma2,self.W2, stride = self.strd2) - gamma1, self.W2, stride = self.strd2)) + self.b2) 
            gamma3 = F.relu( (gamma3 - F.conv2d( F.conv_transpose2d(gamma3,self.W3, stride = self.strd3) - gamma2, self.W3, stride = self.strd3)) + self.b3) 

        
        # classifier
        gamma = gamma3.view(gamma3.shape[0],gamma3.shape[1]*gamma3.shape[2]*gamma3.shape[3])
        out = self.Wclass(gamma)
        out = F.log_softmax(out,dim = 1)
        
        
        return gamma, out
    
    
    
    
####################################################################################################    
    

##################################################

####        MultiLayer J-ISTA with Soft Threshold        ####

##################################################

class ML_JISTA_ST(nn.Module):
    def __init__(self,m1,m2,m3):
        super(ML_JISTA_ST, self).__init__()
        
        # Convolutional Filters
        self.W1 = nn.Parameter(torch.randn(m1,1,6,6), requires_grad=True)
        self.strd1 = 2;
        self.W2 = nn.Parameter(torch.randn(m2,m1,6,6), requires_grad=True)
        self.strd2 = 2;
        self.W3 = nn.Parameter(torch.randn(m3,m2,4,4), requires_grad=True)
        self.strd3 = 1;
        
        # Biases / Thresholds
        self.b1 = nn.Parameter(torch.zeros(1,m1,1,1), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1,m2,1,1), requires_grad=True)
        self.b3 = nn.Parameter(torch.zeros(1,m3,1,1), requires_grad=True)
        
        # Classifier
        self.Wclass = nn.Linear(m3, 10)
        
        # Initialization
        self.W1.data = 0.01 * self.W1.data
        self.W2.data = 0.01 * self.W2.data
        self.W3.data = 0.01 * self.W3.data
        

    
    def forward_joint(self, x, b_y,T=0,RHO=1):        
        # Encoding
        gamma1 = F.relu(F.conv2d(x,self.W1, stride = self.strd1) + self.b1)  
        gamma2 = F.relu(F.conv2d(gamma1,self.W2, stride = self.strd2) + self.b2) 
        X = F.conv2d(gamma2,self.W3, stride = self.strd3)
        gamma3 = torch.zeros(X.shape)
        if X.device.type=='cuda': gamma3 = gamma3.cuda()
        for i in range(10):
            n_ci = float(torch.sum(b_y==i))
            if n_ci>0:
                X_i = X[b_y==i,:,:,:]
                x_i = torch.sqrt(torch.sum(X_i**2,dim=0)) # computing norm of rows
                a_i = F.relu(x_i + 1*self.b3 ) 
                X_i = X_i/x_i
                Z_i = X_i * a_i
                gamma3[b_y==i,:,:,:] = Z_i
     
        for _ in  range(T):
            
            # backward computatoin
            gamma2_ml = F.conv_transpose2d(gamma3,self.W3, stride=self.strd3)
            gamma1_ml = F.conv_transpose2d(gamma2_ml,self.W2, stride=self.strd2)
            
            gamma1 = (1-RHO) * gamma1 + RHO * gamma1_ml
            gamma2 = (1-RHO) * gamma2 + RHO * gamma2_ml
            
            # forward computation
            gamma1 = F.relu( (gamma1 - F.conv2d( F.conv_transpose2d(gamma1,self.W1, stride = self.strd1) - x ,self.W1, stride = self.strd1)) + self.b1)
            gamma2 = F.relu( (gamma2 - F.conv2d( F.conv_transpose2d(gamma2,self.W2, stride = self.strd2) - gamma1, self.W2, stride = self.strd2)) + self.b2) 
            
            X = (gamma3 - F.conv2d( F.conv_transpose2d(gamma3,self.W3, stride = self.strd3) - gamma2, self.W3, stride = self.strd3))
            gamma3 = gamma3*0
            for i in range(10):
                n_ci = float(torch.sum(b_y==i))
                if n_ci>0:
                    X_i = X[b_y==i,:,:,:]
                    x_i = torch.sqrt(torch.sum(X_i**2,dim=0)) # computing norm of rows 
                    a_i = F.relu(x_i + self.b3 ) # n_ci 
                    X_i = X_i/x_i
                    X_i = X_i * a_i
                    gamma3[b_y==i,:,:,:] = Z_i
        
        # classifier
        gamma = gamma3.view(gamma3.shape[0],gamma3.shape[1]*gamma3.shape[2]*gamma3.shape[3])
        out = self.Wclass(gamma)
        out = F.log_softmax(out,dim = 1)
    
        return gamma, out
    
     
    
    def forward(self, x,T=0,RHO=1):
        
        # Encoding
        gamma1 = F.relu(F.conv2d(x,self.W1, stride = self.strd1) + self.b1)      
        gamma2 = F.relu(F.conv2d(gamma1,self.W2, stride = self.strd2) + self.b2) 
        gamma3 = F.relu(F.conv2d(gamma2,self.W3, stride = self.strd3) + self.b3)
        
        for _ in  range(T):
            
            # backward computatoin
            gamma2_ml = F.conv_transpose2d(gamma3,self.W3, stride=self.strd3)
            gamma1_ml = F.conv_transpose2d(gamma2_ml,self.W2, stride=self.strd2)
            
            gamma1 = (1-RHO) * gamma1 + RHO * gamma1_ml
            gamma2 = (1-RHO) * gamma2 + RHO * gamma2_ml
            
            # forward computation
            gamma1 = F.relu( (gamma1 - F.conv2d( F.conv_transpose2d(gamma1,self.W1, stride = self.strd1) - x ,self.W1, stride = self.strd1)) + self.b1)
            gamma2 = F.relu( (gamma2 - F.conv2d( F.conv_transpose2d(gamma2,self.W2, stride = self.strd2) - gamma1, self.W2, stride = self.strd2)) + self.b2) 
            gamma3 = F.relu( (gamma3 - F.conv2d( F.conv_transpose2d(gamma3,self.W3, stride = self.strd3) - gamma2, self.W3, stride = self.strd3)) + self.b3) 

        
        # classifier
        gamma = gamma3.view(gamma3.shape[0],gamma3.shape[1]*gamma3.shape[2]*gamma3.shape[3])
        out = self.Wclass(gamma)
        out = F.log_softmax(out,dim = 1)
        
        
        return gamma, out
    
    
    
    
####################################################################################################   


