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
		max_IHT_iters = 20
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

class SL_CSC_IHT_Joint(nn.Module):
	def __init__(self, stride=1, dp_channels=1, atom_r=1, atom_c=1, numb_atom=1, k=1, alpha=0.1):
		super(SL_CSC_IHT_Joint, self).__init__()
		self.D_trans = nn.Conv2d(dp_channels, numb_atom, (atom_r, atom_c), stride, padding=0, dilation=1, groups=1, bias=False)
		self.D = nn.ConvTranspose2d(numb_atom, dp_channels, (atom_c, atom_r), stride, padding=0, output_padding=0, groups=1, bias=False, dilation=1)
		self.normalise_weights()
		self.D_trans.weight.data = self.D.weight.data.permute(0,1,3,2)
		self.forward_type = 'IHT_Joint'
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
		max_IHT_iters = 20
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

	def forward_training(self, Y, T, p):
		print("Running Joint IHT, projecting onto support cardinality k = {0:1.0f}".format(self.k))
		# Define IHT parameters
		alpha = 0.2 #0.005
		max_IHT_iters = 30
		# Extract data and weight filter dimensions
		y_dims = list(Y.data.size())
		filter_dims = list(self.D_trans.weight.data.size())
		# Order the data so that member fom the same set are processed at the same time
		temp = {"0":[], "1":[], "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
		Z = {}
		X = {}
		Z_list = []
		X_list = []
		error = {}
		total_l2_error = 0
		error_percent= {}
		joint_supp = {}
		numb_runs = {}
		# Put data into into lists of tensors by label
		for i in range(y_dims[0]):
			temp[str(T[i].item())].append(Y[i])
		# Take list of tensors and form stacked tensor, process as go along
		for key, tensor_list in temp.items():
			if len(temp[key]) > 0:
				Z[key] = torch.stack(temp[key], dim=0)
				input_dims = list(Z[key].size())
				active_filter_inds = sf.sample_filters(filter_dims[0], p, self.k)
				self.mask = sf.create_dropout_mask(input_dims[0], filter_dims[0], (input_dims[2]-filter_dims[2]+1), (input_dims[3]-filter_dims[3]+1), active_filter_inds)
				X[key], error_percent[key], error[key], numb_runs[key], joint_supp[key] = self.IHT_Joint(Z[key], max_IHT_iters, alpha)
				total_l2_error = total_l2_error + error[key]
				X_list.append(X[key])
				Z_list.append(Z[key])
		X_tensor = torch.cat(X_list, dim=0)
		Z_tensor = torch.cat(Z_list, dim=0)
		total_error_percent = total_l2_error*100/(np.sum((Y).data.cpu().numpy()**2))
		av_number_runs=0
		for key, sc_iterations in numb_runs.items():
			av_number_runs = av_number_runs+sc_iterations
		av_number_runs = av_number_runs/len(numb_runs)
		filters_selected = []
		for key, js in joint_supp.items():
			filters_selected = np.append(filters_selected, js)
		
		print("Error breakdown by class label:")
		for key, item in error_percent.items():
			print("Class: {}".format(key) + ", number of iterations: {}".format(numb_runs[key]) + ", error percentage: {0:1.2f}%".format(item))
		return X_tensor, Z_tensor, total_error_percent, av_number_runs, joint_supp, filters_selected, X
				

	def IHT_Joint(self, Y, max_IHT_iters, alpha):
		# print("Running IHT, projecting onto support cardinality k = {0:1.0f}".format(self.k))
		y_dims = list(Y.data.size())
		w_dims = list(self.D_trans.weight.data.size())
		# Initialise X as zero tensor
		X1 = Variable(torch.zeros(y_dims[0], w_dims[0], (y_dims[2]-w_dims[2]+1),(y_dims[3]-w_dims[3]+1)).to(device, dtype = dtype))
		X1_error = np.sum((Y).data.cpu().numpy()**2)
		X2_error = 0
		i=0
		run = True
		while run == True and i< max_IHT_iters:
			g = self.dropout(self.D_trans(Y-self.D(self.dropout(X1))))
			HT_arg = X1 + self.alpha*g
			X2, filters_selected = sf.hard_threshold_joint_k(HT_arg, self.k)
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
				# print("After " +repr(i+1) + " iterations of IHT, average l2 error over batch: {0:1.2f}".format(error_percent) + "% , Av. sparsity per image: {0:1.2f}".format(percent_zeros_per_image) +"%")
			i=i+1
		return X1, error_percent, l2_error, i, filters_selected


	def normalise_weights(self):
		filter_dims = list(np.shape(self.D.weight.data.cpu().numpy()))
		for i in range(filter_dims[0]):
			for j in range(filter_dims[1]):
				self.D.weight.data[i][j] = self.D.weight.data[i][j]/((np.sum(self.D.weight.data[i][j].cpu().numpy()**2))**0.5)

	def dropout(self,X):
		X_dropout = Variable(X.data*self.mask)
		return X_dropout

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





