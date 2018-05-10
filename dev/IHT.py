import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

import random
import os
import yaml

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

import AuxiliaryFunctions as af
import SupportingFunctions as sf 



# Define hardware
use_cuda = True
can_use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if can_use_cuda else "cpu")
# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.float
dtype = torch.float
using_azure = False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SL_CSC_IHT(nn.Module):
	def __init__(self, stride=1, dp_channels=1, atom_r=1, atom_c=1, numb_atom=1, k=1, alpha=0.1):
		super(SL_CSC_IHT, self).__init__()
		self.D_trans = nn.Conv2d(dp_channels, numb_atom, (atom_r, atom_c), stride, padding=0, dilation=1, groups=1, bias=False)
		# self.dropout = nn.Dropout2d(p=0.5, inplace=False)
		self.D = nn.ConvTranspose2d(numb_atom, dp_channels, (atom_c, atom_r), stride, padding=0, output_padding=0, groups=1, bias=False, dilation=1)
		self.normalise_weights()
		self.D_trans.weight.data = self.D.weight.data.permute(0,1,3,2)
		self.forward_type = 'IHT'
		self.batch_size = 1
		self.k=k
		self.mask = torch.ones(self.batch_size, numb_atom, atom_r, atom_c).to(dtype=dtype)
		self.alpha = alpha


	def forward(self, Y):
		print("Running IHT, projecting onto support cardinality k = {0:1.0f}".format(self.k))
		y_dims = list(Y.data.size())
		w_dims = list(self.D_trans.weight.data.size())
		# Initialise X as zero tensor
		X1 = Variable(torch.zeros(y_dims[0], w_dims[0], (y_dims[2]-w_dims[2]+1),(y_dims[3]-w_dims[3]+1)).to(device, dtype = dtype))
		alpha = 0.2 #0.005 # Delete after testing
		X1_error = np.sum((Y).data.cpu().numpy()**2)
		X2_error = 0
		i=0
		max_IHT_iters = 30
		run = True
		while run == True and i< max_IHT_iters:
			g = self.dropout(self.D_trans(Y-self.D(self.dropout(X1))))
			HT_arg = X1 + self.alpha*g
			X2, filters_selected = sf.hard_threshold_k(HT_arg, self.k)
			X2_error = np.sum(((Y-self.D(self.dropout(X2))).data.cpu().numpy())**2)
			if X2_error < X1_error:
				X1 = X2
				X1_error = X2_error
			else:
				run = False
			if i==0 or (i+1)%1 == 0:
				# After run IHT print out the result
				l2_error = X1_error
				av_num_zeros_per_image = X1.data.nonzero().cpu().numpy().shape[0]/y_dims[0]
				percent_zeros_per_image = 100*av_num_zeros_per_image/(y_dims[2]*y_dims[3])
				error_percent = l2_error*100/(np.sum((Y).data.cpu().numpy()**2))
				print("After " +repr(i+1) + " iterations of IHT, average l2 error over batch: {0:1.2f}".format(error_percent) + "% , Av. sparsity per image: {0:1.2f}".format(percent_zeros_per_image) +"%")
			i=i+1
		return X1, error_percent, i, filters_selected


	def normalise_weights(self):
		filter_dims = list(np.shape(self.D.weight.data.cpu().numpy()))
		for i in range(filter_dims[0]):
			for j in range(filter_dims[1]):
				self.D.weight.data[i][j] = self.D.weight.data[i][j]/((np.sum(self.D.weight.data[i][j].cpu().numpy()**2))**0.5)

	def dropout(self,X):
		X_dropout = Variable(X.data*self.mask)
		return X_dropout

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






