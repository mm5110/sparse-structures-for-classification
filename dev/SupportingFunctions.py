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
dtype = torch.float
using_azure = True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SUPPORTING ALGORITHM FUNCTIONS
def soft_thresh(x, alpha):
	# x is a pytorch variable, extract weights tensor and convert to a numpy array
	x_numpy = x.data.numpy()
	# Apply soft thresholding function to numpy array 
	z = np.absolute(x_numpy) - alpha
	z[z<0] = 0
	y_numpy = np.multiply(z, np.sign(x_numpy))
	# Convert resulting numpy array back to a Pytorch Variable and return it
	y = Variable(torch.from_numpy(y_numpy)).type(torch.FloatTensor)
	return y

def hard_threshold_k(X, k):
	Gamma = X.clone()
	Gamma = Gamma.view(Gamma.data.shape[0], Gamma.data.shape[1]*Gamma.data.shape[2]*Gamma.data.shape[3])
	m = X.data.shape[1]
	a,_ = torch.abs(Gamma).data.sort(dim=1,descending=True)
	T = torch.mm(a[:,k].unsqueeze(1),torch.Tensor(np.ones((1,m))))
	mask = Variable(torch.Tensor( (np.abs(Gamma.data.numpy())>T.numpy()) + 0.))
	Gamma = Gamma * mask
	Gamma = Gamma.view(X.data.shape)
	return Gamma, mask.data.nonzero()

def project_onto_sup(X, sup):
	X_numpy = X.data.numpy()
	X_dims = list(np.shape(X_numpy))
	X_new = np.zeros((X_dims[0], X_dims[1], X_dims[2], X_dims[3]))
	for i in range(X_dims[0]):
		for j in range(len(supp)):
			X_new[i][sup[j][0]][sup[j][1]][sup[j][2]] = X_numpy[i][sup[j][0]][sup[j][1]][sup[j][2]]
	X_out = Variable(torch.from_numpy(X_new).type(torch.FloatTensor))
	return X_out

def sample_filters(numb_atoms, p, k):
	numb_active_filters = int(np.maximum(np.ceil(p*numb_atoms), k))
	active_filter_inds = random.sample(range(0, numb_atoms), numb_active_filters)
	return active_filter_inds

def create_dropout_mask(numb_dp, numb_atoms, numb_r, numb_c, active_filter_inds):
	temp = torch.zeros(numb_atoms, numb_r, numb_c)
	for i in active_filter_inds:
		temp[i] = torch.ones(numb_r, numb_c)

	temp = torch.unsqueeze(temp,0)
	mask = temp.repeat(numb_dp,1,1,1)
	return mask

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PRIMARY ALGORITHM FUNCTIONS
def train_SL_CSC(CSC, train_loader, num_epochs, T_DIC, cost_function, CSC_parameters, learning_rate, momentum, weight_decay, batch_size, p):	
	print("Training SL-CSC. Batch size is: " + repr(batch_size))
	# Define optimizer
	optimizer = torch.optim.Adam(CSC_parameters, lr=learning_rate,weight_decay=weight_decay)
	# Initialise variables needed to plot a random sample of three kernels as they are trained
	filter_dims = list(np.shape(CSC.D_trans.weight.data.numpy()))
	idx = random.sample(range(0, filter_dims[0]), 3)
	# Prepare logging files and data
	run_code = 1
	fieldnames = ['Epoch', 'Batch Number', 'Total Batch Number', 'l2 SC', 'Number SC Iterations', 'l2 End', 'Average Sparsity', 'Cumulative Filters Trained']
	initialise = True
	training_log_filename = 'log_data/' + str(run_code) + '_' + str(CSC.forward_type) + '_' + 'training_log.csv'
	activation_data_filename = 'log_data/' + str(run_code) + '_' + str(CSC.forward_type) + '_' + 'activations'
	filter_activations = np.zeros(filter_dims[0])
	# Variables for plotting error as go along
	counter = 1
	l2_error_list = np.empty(0)	
	# Prepare plots of filters
	plt.ion()
	plt.show()
	for epoch in range(num_epochs):
		print("Training epoch " + repr(epoch+1) + " of " + repr(num_epochs))
		for i, (inputs, labels) in enumerate(train_loader):
			print("Batch number " + repr(i+1))
			inputs = Variable(inputs).to(device)
			labels = Variable(labels).to(device)
			# Calculate and update step size for sparse coding step
			input_dims = list(inputs.size())
			CSC.batch_size = input_dims[0]
			# Generate dropout filter
			active_filter_inds = sample_filters(filter_dims[0], p, CSC.k)
			CSC.mask = create_dropout_mask(input_dims[0], filter_dims[0], (input_dims[2]-filter_dims[2]+1), (input_dims[3]-filter_dims[3]+1), active_filter_inds)
			# Fix dictionary and calculate sparse code
			if CSC.forward_type == 'FISTA_fixed_step':
				CSC.calc_L(input_dims)
			X, SC_error_percent, numb_SC_iterations, filters_selected = CSC.forward(inputs)
			X = X.detach()
			# Update filter activations
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
			average_number_nonzeros = X.data.nonzero().numpy().shape[0]/input_dims[0]
			# Fix sparse code and update dictionary
			print("Running dictionary update")
			for j in range(T_DIC):
				# Zero the gradient
				optimizer.zero_grad()
				# Calculate estimate of reconstructed Y
				inputs_recon = CSC.reverse(X)
				# Calculate loss according to the defined cost function between the true Y and reconstructed Y
				loss = cost_function(inputs_recon, inputs)
				# Calculate the gradient of the cost function wrt to each parameters
				loss.backward()
				# Update each parameter according to the optimizer update rule (single step)
				optimizer.step()
				# At the end of each batch plot a random sample of kernels to observe progress
				if j==0 or (j+1)%5 == 0:
					print("Average loss per data point at iteration {0:1.0f}".format(j+1) + " of SGD: {0:1.4f}".format(np.asscalar(loss.data.numpy())))
					if using_azure == False:
						plt.figure(11)
						plt.clf()
						plt.subplot(1,3,1)
						plt.imshow((CSC.D.weight[idx[0]][0].data.numpy()), cmap='gray')
						plt.title("Filter "+repr(idx[0]))
						plt.subplot(1,3,2)
						plt.imshow((CSC.D.weight[idx[1]][0].data.numpy()), cmap='gray', )
						plt.title("Filter "+repr(idx[1]))
						plt.xlabel("Epoch Number: " + repr(epoch)+ ", Batch number: " + repr(i+1) + ", Average loss: {0:1.4f}".format(np.asscalar(loss.data.numpy())))
						plt.subplot(1,3,3)
						plt.imshow((CSC.D.weight[idx[2]][0].data.numpy()), cmap='gray')
						plt.title("Filter "+repr(idx[2]))
						plt.draw()
						plt.pause(0.001)
			# Calculate l2 error
			l2_error_percent = 100*np.sum((inputs-CSC.D(X)).data.numpy()**2)/ np.sum((inputs).data.numpy()**2)
			l2_error_list = np.append(l2_error_list, l2_error_percent)
			print("After " +repr(j+1) + " iterations of SGD, average l2 error over batch: {0:1.2f}".format(l2_error_percent) + "%")
			# Plot training error overtime
			if using_azure == False:
				plt.figure(12)
				plt.clf()
				plt.plot(np.arange(counter), l2_error_list)
				plt.title("Training error over time")
				plt.ylabel("Percentage error")
				plt.xlabel("Number of batches")
				plt.draw()
				plt.pause(0.001)
			# Normalise each atom / kernel
			CSC.normalise_weights()
			# Ensure that weights for the reverse and forward operations are consistent	
			CSC.D_trans.weight.data = CSC.D.weight.data.permute(0,1,3,2)
			# Reset optimizer
			optimizer = torch.optim.Adam(CSC_parameters, lr=learning_rate,weight_decay=weight_decay)
			# Log training data
			log_data = [epoch, i, counter, SC_error_percent, numb_SC_iterations, l2_error_percent, average_number_nonzeros, np.count_nonzero(filter_activations)]
			af.log_training_data(training_log_filename, initialise, log_data, fieldnames)
			initialise = False
			# Update counter
			counter=counter+1
	# Reset the mask for non training state (i.e. no dropout)
	CSC.mask = torch.ones(input_dims[0], filter_dims[0], (input_dims[2]-filter_dims[2]+1), (input_dims[3]-filter_dims[3]+1))
	# Save down the filter activations
	np.save(activation_data_filename, filter_activations)
	return CSC
