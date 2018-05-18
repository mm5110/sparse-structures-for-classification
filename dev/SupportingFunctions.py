import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

import random
import os
import yaml
import datetime
import csv

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


# Define hardware
use_cuda = True
can_use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if can_use_cuda else "cpu")
# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.float
dtype = torch.float
using_azure = False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SUPPORTING ALGORITHM FUNCTIONS
def hard_threshold_joint_k(X, k):
	Gamma = X.clone()
	Gamma = Gamma.view(Gamma.data.shape[0], Gamma.data.shape[1]*Gamma.data.shape[2]*Gamma.data.shape[3])
	filter_activation_l2 = np.sum(Gamma.data.numpy()**2, axis=0)
	joint_supp =  np.argsort(filter_activation_l2)[-k:]
	mask = torch.zeros(Gamma.data.shape[0], Gamma.data.shape[1]).to(device, dtype=dtype)
	for i in range(k):
		mask[:,joint_supp[i]] = torch.ones(Gamma.data.shape[0])
	Gamma = Gamma * mask
	Gamma = Gamma.view(X.data.shape)
	return Gamma, joint_supp

def hard_threshold_k(X, k):
	Gamma = X.clone()
	Gamma = Gamma.view(Gamma.data.shape[0], Gamma.data.shape[1]*Gamma.data.shape[2]*Gamma.data.shape[3])
	m = X.data.shape[1]
	a,_ = torch.abs(Gamma).data.sort(dim=1,descending=True)
	T = torch.mm(a[:,k].unsqueeze(1),torch.Tensor(np.ones((1,m))).to(device, dtype=dtype))
	mask = Variable(torch.Tensor((np.abs(Gamma.data.cpu().numpy())>T.cpu().numpy()) + 0.)).to(device, dtype=dtype)
	Gamma = Gamma * mask
	Gamma = Gamma.view(X.data.shape)
	return Gamma, mask.data.nonzero()

def sample_filters(numb_atoms, p, k):
	numb_active_filters = int(np.maximum(np.ceil(p*numb_atoms), k))
	active_filter_inds = random.sample(range(0, numb_atoms), numb_active_filters)
	return active_filter_inds

def create_dropout_mask(numb_dp, numb_atoms, numb_r, numb_c, active_filter_inds):
	temp = torch.zeros(numb_atoms, numb_r, numb_c).to(dtype=dtype)
	for i in active_filter_inds:
		temp[i] = torch.ones(numb_r, numb_c)
	temp = torch.unsqueeze(temp,0)
	mask = temp.repeat(numb_dp,1,1,1)
	mask = mask.to(device, dtype=dtype)
	return mask

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PRIMARY ALGORITHM FUNCTIONS
def train_SL_CSC(CSC, train_loader, test_loader, num_epochs, T_DIC, cost_function, CSC_parameters, learning_rate, momentum, weight_decay, batch_size, p, model_filename):	
	print("Training SL-CSC. Batch size is: " + repr(batch_size))
	# Define optimizer
	optimizer = torch.optim.Adam(CSC_parameters, lr=learning_rate, weight_decay=weight_decay)
	# Initialise variables needed to plot a random sample of three kernels as they are traineds
	filter_dims = list(np.shape(CSC.D_trans.weight.data.cpu().numpy()))
	# idx = random.sample(range(0, filter_dims[0]), 3)
	# Prepare logging files and data
	fieldnames = ['Epoch', 'Batch Number', 'Total Batch Number', 'l2 SC', 'Number SC Iterations', 'l2 End', 'Average Sparsity', 'Cumulative Filters Trained']
	initialise = True
	training_log_filename = 'log_data/' + model_filename + '_' + 'training_log.csv'
	activation_data_filename = 'log_data/' + model_filename  + '_' + 'activations'
	filter_activations = np.zeros(filter_dims[0])
	# Variables for plotting training and validation error as going along
	counter = 1
	l2_error_list = np.empty(0)
	sc_error_list = np.empty(0)
	test_l2_error_list =  np.empty(0)
	test_sc_error_list = np.empty(0)
	test_error_xaxis = np.empty(0)
	validation_run = 10
	# Variables to control learning rate
	mva_numb = 10
	beta = 0.98
	min_error = 0.5
	min_learning_rate = 0.00001
	min_alpha = 0.01
	# Prepare plots of filters
	plt.ion()
	plt.show()
	# Begin training loop
	for epoch in range(num_epochs):
		print("Training epoch " + repr(epoch+1) + " of " + repr(num_epochs))
		for i, (inputs, labels) in enumerate(train_loader):
			print("Batch number " + repr(i+1))
			inputs = Variable(inputs).to(device)
			# Remove the average of the batch
			inputs_mean = torch.mean(inputs.data, dim=0, keepdim=True)
			inputs = inputs - inputs_mean
			labels = Variable(labels).to(device)
			# Calculate and update step size for sparse coding step
			input_dims = list(inputs.size())
			CSC.batch_size = input_dims[0]
			# Fix dictionary and calculate sparse code
			if CSC.forward_type == 'IHT_Joint':
				print("Forward type IHT_Joint")
				X, inputs, SC_error_percent, numb_SC_iterations, joint_supp, filters_selected = CSC.forward_training(inputs, labels, p)
				inputs.detach()
			else:
				print("Forward type IHT")
				# Generate dropout filter
				active_filter_inds = sample_filters(filter_dims[0], p, CSC.k)
				CSC.mask = create_dropout_mask(input_dims[0], filter_dims[0], (input_dims[2]-filter_dims[2]+1), (input_dims[3]-filter_dims[3]+1), active_filter_inds)
				X, SC_error_percent, numb_SC_iterations, filters_selected = CSC.forward(inputs)
			X = X.detach()
			sc_error_list = np.append(sc_error_list, SC_error_percent)
			if counter > mva_numb:
				mva_sc_l2_error = np.sum(sc_error_list[-mva_numb:])/mva_numb
				if np.abs(SC_error_percent - mva_sc_l2_error) < min_error:
					CSC.alpha = max(beta*CSC.alpha, min_alpha)
					print("Training error has not decreased significantly, alpha reduced to: {0:1.2f}".format(CSC.alpha))
			# Update filter activations
			if CSC.forward_type == 'IHT_Joint': 
				for l in range(len(filters_selected)):
					filter_activations[int(filters_selected[l])] = filter_activations[int(filters_selected[l])] + 1
			else:
				for l in range(len(filters_selected)):
					filter_activations[filters_selected[l][1]] = filter_activations[filters_selected[l][1]] + 1
			# Plot historgram of filter activations
			if using_azure == False:
				plt.figure(10)
				plt.clf()
				plt.hist(filter_activations, bins=50)  # arguments are passed to np.histogram
				plt.title("Histogram of filter activations, epoch Number: {0:1.0f}".format(epoch) + ", batch number: {0:1.0f}".format(i+1))
				plt.ylabel("Number of filters")
				plt.xlabel("Activation frequency bin")
				plt.draw()
				plt.pause(0.001)
			# Calculate the average number of zeros per data input
			average_number_nonzeros = X.data.nonzero().cpu().numpy().shape[0]/input_dims[0]
			# Fix sparse code and update dictionary
			print("Running dictionary update")
			for j in range(T_DIC):
				# Zero the gradient
				optimizer.zero_grad()
				# Calculate estimate of reconstructed Y
				inputs_recon = CSC.D(X)
				# Calculate loss according to the defined cost function between the true Y and reconstructed Y
				loss = cost_function(inputs_recon, inputs)
				# Calculate the gradient of the cost function wrt to each parameters
				loss.backward()
				# Update each parameter according to the optimizer update rule (single step)
				optimizer.step()
				# At the end of each batch plot a random sample of kernels to observe progress
				if j==0 or (j+1)%10 == 0:
					print("Average loss per data point at iteration {0:1.0f}".format(j+1) + " of SGD: {0:1.4f}".format(np.asscalar(loss.data.cpu().numpy())))
					# if using_azure == False:
					# 	D = CSC.D.weight.data.cpu().numpy()
					# 	M = af.showFilters(D,10,10)
					# 	plt.figure(11, figsize=(10,10))
					# 	plt.imshow(rescale(M, scale=4, mode='constant'),cmap='gray')
					# 	plt.axis('off')
					# 	plt.draw()
					# 	plt.pause(0.001)
			# Calculate l2 training error
			l2_error_percent = 100*np.sum((inputs-CSC.D(X)).data.cpu().numpy()**2)/ np.sum((inputs).data.cpu().numpy()**2)
			l2_error_list = np.append(l2_error_list, l2_error_percent)
			print("After " +repr(j+1) + " iterations of SGD, average l2 error over batch: {0:1.2f}".format(l2_error_percent) + "%")
			# Run validation
			if counter%validation_run==0:
				print("Running on validation set")
				test_inputs, test_labels = next(iter(test_loader))
				test_inputs = Variable(inputs).to(device)
				test_labels = Variable(labels).to(device)
				test_input_dims = list(test_inputs.size())
				CSC.mask = torch.ones(test_input_dims[0], filter_dims[0], (input_dims[2]-filter_dims[2]+1), (input_dims[3]-filter_dims[3]+1))			
				test_X, test_SC_error_percent, test_numb_SC_iterations, test_filters_selected = CSC.forward(test_inputs)
				test_l2_error_percent = 100*np.sum((inputs-CSC.D(test_X)).data.cpu().numpy()**2)/ np.sum((test_inputs).data.cpu().numpy()**2)
				test_l2_error_list = np.append(test_l2_error_list, test_l2_error_percent)
				test_error_xaxis = np.append(test_error_xaxis, counter)			
			# Plot training error overtime
			if using_azure == False:
				plt.figure(12)
				plt.clf()
				plt.plot(np.arange(counter), l2_error_list, label='Training error')
				plt.plot(test_error_xaxis, test_l2_error_list, label='Validation error')
				plt.title("Reconstruction error over time")
				plt.ylabel("Percentage reconstruction error over batch")
				plt.xlabel("Number of batches")
				plt.legend()
				plt.draw()
				plt.pause(0.001)
			# Normalise each atom / kernel
			CSC.normalise_weights()
			# Ensure that weights for the reverse and forward operations are consistent	
			CSC.D_trans.weight.data = CSC.D.weight.data.permute(0,1,3,2)
			# Adjust learning rate if is learning to slowly
			if counter > mva_numb:
				mva_l2_error = np.sum(l2_error_list[-mva_numb:])/mva_numb
				if np.abs(l2_error_percent - mva_l2_error) < min_error:
					learning_rate = max(beta*learning_rate, min_learning_rate)
					print("Training error has not decreased significantly, learning rate reduced to: {0:1.5f}".format(learning_rate))
			# Reset optimizer
			optimizer = torch.optim.Adam(CSC_parameters, lr=learning_rate,weight_decay=weight_decay)
			# Log training data
			log_data = [epoch, i, counter, SC_error_percent, numb_SC_iterations, l2_error_percent, average_number_nonzeros, np.count_nonzero(filter_activations)]
			af.log_training_data(training_log_filename, initialise, log_data, fieldnames)
			initialise = False
			# Update counter
			counter=counter+1
	# Save down the filter activations
	np.save(activation_data_filename, filter_activations)
	return CSC
