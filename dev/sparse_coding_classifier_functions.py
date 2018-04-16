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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SAVE AND LOAD FUNCTIONS
def save_SLCSC_FISTA(CSC, stride, dp_channels, atom_r, atom_c, numb_atom, filename):
	torch_save_path = os.getcwd() + "/trained_models/" + filename + ".pt"
	yml_save_path = os.getcwd() + "/trained_models/" + filename + ".yml"
	# Save model parameters
	torch.save(CSC.state_dict(), torch_save_path)
	# Define dictionary of other variables to store
	other_CSC_variable_data = {}
	other_CSC_variable_data["stride"] = stride
	other_CSC_variable_data["dp_channels"] = dp_channels
	other_CSC_variable_data["atom_r"] = atom_r
	other_CSC_variable_data["atom_c"] = atom_c
	other_CSC_variable_data["numb_atom"] = numb_atom
	other_CSC_variable_data["tau"] = CSC.tau
	other_CSC_variable_data["step_size"] = CSC.step_size
	other_CSC_variable_data["T_SC"] = CSC.T_SC
	other_CSC_variable_data["T_PM"] = CSC.T_PM
	# Save down dictionary in a yaml file
	with open(yml_save_path, 'w') as yaml_file:
		yaml.dump(other_CSC_variable_data, stream=yaml_file, default_flow_style=False)


def load_SLCSC_FISTA(filename):
	torch_load_path = os.getcwd() + "/trained_models/" + filename + ".pt"
	yml_load_path = os.getcwd() + "/trained_models/" + filename + ".yml"

	# Load in model
	with open(yml_load_path, 'r') as yaml_file:
		loaded_CSC_vars = yaml.load(yaml_file)

	# Initialise and return CSC
	CSC = SL_CSC_FISTA(loaded_CSC_vars["stride"], loaded_CSC_vars["dp_channels"], loaded_CSC_vars["atom_r"], loaded_CSC_vars["atom_c"], loaded_CSC_vars["numb_atom"], loaded_CSC_vars["tau"], loaded_CSC_vars["T_SC"], loaded_CSC_vars["T_PM"], loaded_CSC_vars["step_size"])
	# Load in network parameters
	CSC.load_state_dict(torch.load(torch_load_path))
	# Return model 
	return CSC

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SUPPORTING ALGORITHM FUNCTIONS
def soft_thresh(x, alpha):
	# x is a pytorch variable, extract weights tensor and convert to a numpy array
	x_numpy = x.data.numpy()
	# Apply soft thresholding function to numpy array 
	z = np.absolute(x_numpy) - alpha
	z[z<0] = 0
	y_numpy = np.multiply(z, np.sign(x_numpy))
	# Convert resulting numpy array back to a Pytorch Variable and return it
	y = Variable(torch.from_numpy(y_numpy))
	return y

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PRIMARY ALGORITHM FUNCTIONS
def train_SL_CSC(CSC, train_loader, num_epochs, T_DIC, cost_function, optimizer):	
	print("Training SL-CSC. Batch size is: ")
	for epoch in range(num_epochs):
		print("Training epoch " + repr(epoch+1) + "of " + repr(num_epochs))
		for i, (inputs, labels) in enumerate(train_loader):
			print("Batch number " + repr(i+1))
			inputs = Variable(inputs)
			labels = Variable(labels)
			# Calculate and update step size for sparse coding step
			input_dims = list(inputs.size())
			CSC.calc_step_size(input_dims)
			# Fix dictionary and calculate sparse code
			X = CSC.forward(inputs)
			# Fix sparse code and update dictionary
			print("Running dictionary update")
			# Update weight matrix
			for i in range(T_DIC):
				# Zero the gradient
				optimizer.zero_grad()
				# Calculate estimate of reconstructed Y
				inputs_recon = CSC.reverse(X)
				# Calculate loss according to the defined cost function between the true Y and reconstructed Y
				loss = cost_function(inputs_recon, inputs)
				if (i+1)%20 == 0:
					print("Average loss per data point at iteration " +repr(i+1) + " :" + repr(np.asscalar(loss.data.numpy())))
				# Calculate the gradient of the cost function wrt to each parameters
				loss.backward()
				# Update each parameter according to the optimizer update rule (single step)
				optimizer.step()
			# Ensure that weights for the CL-CSC and inverse CL-CSC are consistent	
			CSC.D_trans.weight.data = CSC.D.weight.data.permute(0,1,3,2)
			# Update the step size value based off the last weights update
			CSC.calc_step_size(input_dims)
	return CSC

		
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CSC CLASSES AND CONSISTENCY FUNCTIONS
class SL_CSC_FISTA(nn.Module):
	def __init__(self, stride=1, dp_channels=1, atom_r=1, atom_c=1, numb_atom=1, tau=1, T_SC=1, T_PM=1, step_size=1):
		super(SL_CSC_FISTA, self).__init__()
		self.D_trans = nn.Conv2d(dp_channels, numb_atom, (atom_r, atom_c), stride, padding=0, dilation=1, groups=1, bias=False)
		self.D = nn.ConvTranspose2d(numb_atom, dp_channels, (atom_c, atom_r), stride, padding=0, output_padding=0, groups=1, bias=False, dilation=1)
		self.D_trans.weight.data = self.D.weight.data.permute(0,1,3,2)
		self.tau = tau
		self.step_size = step_size
		self.T_SC = T_SC
		self.T_PM = T_PM
		if tau > 1:
			print("WARNING: regularisation parameter tau is larger than 1, consider reducing")

	def forward(self, Y):
		# Generate latent representation
		# Process according to sparse recovery method
		print("Running FISTA to recover/ estimate sparse code")
		# Initialise t variables needed for FISTA
		t2 = 1
		# Initialise X1 Variable - note we need X1 and X2 as we need to use the prior two prior estimates for each update
		y_dims = list(Y.data.size())
		w_dims = list(self.D_trans.weight.data.size())
		X1 = Variable(torch.randn(y_dims[0], w_dims[0], (y_dims[2]-w_dims[2]+1),(y_dims[3]-w_dims[3]+1)))
		# Minimizer argument
		ST_arg = X1 - self.step_size*self.D_trans(self.D(X1)-Y)
		
		for i in range(0, self.T_SC):
			# Calculate latest sparse code estimate
			X2 = soft_thresh(ST_arg, self.step_size)
			if (i+1)%10 == 0:
				print("Iteration: "+repr(i+1)+ ", Sparsity level: " +repr(X2.data.nonzero().numpy().shape[0]))
			# If this was not the last iteration then update variables needed for the next iteration
			if i < self.T_SC:
				# Update t variables
				t1 = t2
				t2 = (1 + np.sqrt(1+4*(t1**2)))/2 
				# Update Z
				Z = X2 + (X2-X1)*(t1 - 1)/t2
				# Construct minimizer argument to feed into the soft thresholding function
				ST_arg = Z - self.step_size*self.D_trans((self.D(Z)-Y))
				# Update X1 to calculate next Z value
				X1 = X2
		# Return sparse representation
		return X2

	def reverse(self, x):
		out = self.D(x)
		return out
	
	def calc_step_size(self, y_dims):
		print("Calculating Lipschitz constant using power method with " + repr(self.T_PM) + " iterations")
		# Intialise initial and random guess of dominant singular vector
		w_dims = list(self.D_trans.weight.data.size())
		X = Variable(torch.randn(1, w_dims[0], (y_dims[2]-w_dims[2]+1),(y_dims[3]-w_dims[3]+1)))
		# Normalise initialised vector in the l2 norm
		X = X/(np.asscalar(np.sum((X**2).data.numpy()))**0.5)
		# Perform T_PM iterations applying A^TA to x then normalising
		for i in range(self.T_PM):
			X = self.D_trans(self.D(X))
			X = X/(np.asscalar(np.sum((X**2).data.numpy()))**0.5)
		# Compute dominant singular value
		L = (np.asscalar(np.sum((self.D_trans(self.D(X))**2).data.numpy())))**0.5
		self.step_size = self.tau/L
		print("Step size value calculated: " + repr(self.step_size))
		