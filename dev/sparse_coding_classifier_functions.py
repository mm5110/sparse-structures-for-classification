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
def save_SL_CSC_IHT(CSC, stride, dp_channels, atom_r, atom_c, numb_atom, filename):
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
	other_CSC_variable_data["k"] = CSC.k
	other_CSC_variable_data["T_SC"] = CSC.T_SC
	# Save down dictionary in a yaml file
	with open(yml_save_path, 'w') as yaml_file:
		yaml.dump(other_CSC_variable_data, stream=yaml_file, default_flow_style=False)

def load_SL_CSC_IHT(filename):
	torch_load_path = os.getcwd() + "/trained_models/" + filename + ".pt"
	yml_load_path = os.getcwd() + "/trained_models/" + filename + ".yml"
	# Load in model
	with open(yml_load_path, 'r') as yaml_file:
		loaded_CSC_vars = yaml.load(yaml_file)

	# Initialise and return CSC
	CSC = SL_CSC_FISTA(loaded_CSC_vars["stride"], loaded_CSC_vars["dp_channels"], loaded_CSC_vars["atom_r"], loaded_CSC_vars["atom_c"], loaded_CSC_vars["numb_atom"], loaded_CSC_vars["T_SC"], loaded_CSC_vars["k"])
	# Load in network parameters
	CSC.load_state_dict(torch.load(torch_load_path))
	# Return model 
	return CSC

def save_SL_CSC_FISTA(CSC, stride, dp_channels, atom_r, atom_c, numb_atom, filename):
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


def load_SL_CSC_FISTA(filename):
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

def keep_k_largest(X, k):
	# x is a pytorch variable, extract data tensor and convert to a numpy array
	X_numpy = X.data.numpy()
	X_dims = list(np.shape(X_numpy))
	X_new = np.zeros((X_dims[0], X_dims[1], X_dims[2], X_dims[3]))	
	for i in range(X_dims[0]):
		# copy image data, want to sort elements only for a given image
		temp = np.copy(X_numpy[i])
		# extract a list of the ordered elements
		inds_ordered = np.dstack(np.unravel_index(np.argsort(abs(temp).ravel()), (X_dims[1], X_dims[2], X_dims[3])))
		# group all but the k largest indices into a set to be zeroed
		zero_inds = inds_ordered[0][:-k]
		# Zero all but the k largest entries of x
		temp[zero_inds]=0
		# For back into a variable and return
		X_new[i] = temp	
	X_new_var = Variable(torch.from_numpy(X_new).type(torch.FloatTensor))
	return X_new_var


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PRIMARY ALGORITHM FUNCTIONS
def train_SL_CSC(CSC, train_loader, num_epochs, T_DIC, cost_function, optimizer, batch_size):	
	print("Training SL-CSC. Batch size is: " + repr(batch_size))
	# Initialise variables needed to plot a random sample of three kernels as they are trained
	filter_dims = list(np.shape(CSC.D_trans.weight.data.numpy()))
	idx = random.sample(range(0, filter_dims[0]), 3)
	plt.ion()
	plt.show()
	for epoch in range(num_epochs):
		print("Training epoch " + repr(epoch+1) + " of " + repr(num_epochs))
		for i, (inputs, labels) in enumerate(train_loader):
			print("Batch number " + repr(i+1))
			inputs = Variable(inputs)
			labels = Variable(labels)
			# Calculate and update step size for sparse coding step
			input_dims = list(inputs.size())
			if CSC.forward_type == 'FISTA':
				CSC.calc_step_size(input_dims)
			# Fix dictionary and calculate sparse code
			X = CSC.forward(inputs)
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
				if (j+1)%5== 0:
					print("Average loss per data point at iteration " +repr(j+1) + " :" + repr(np.asscalar(loss.data.numpy())))
					plt.figure(1)
					plt.subplot(1,3,1)
					plt.imshow((CSC.D_trans.weight[idx[0]][0].data.numpy()), cmap='gray')
					plt.title("Filter "+repr(idx[0]))
					plt.subplot(1,3,2)
					plt.imshow((CSC.D_trans.weight[idx[1]][0].data.numpy()), cmap='gray', )
					plt.title("Filter "+repr(idx[1]))
					plt.xlabel("Epoch Number: " + repr(epoch)+ ", Batch number: " + repr(i+1) + ", Average loss: {0:1.4f}".format(np.asscalar(loss.data.numpy())))
					plt.subplot(1,3,3)
					plt.imshow((CSC.D_trans.weight[idx[2]][0].data.numpy()), cmap='gray')
					plt.title("Filter "+repr(idx[2]))
					plt.draw()
					plt.pause(0.001)
			# Ensure that weights for the reverse and forward operations are consistent	
			CSC.D_trans.weight.data = CSC.D.weight.data.permute(0,1,3,2)
	# Return trained CSC
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
		self.forward_type = 'FISTA'

		if tau > 1:
			print("WARNING: regularisation parameter tau is larger than 1, consider reducing")

	def forward(self, Y):
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
			if (i+1)%5 == 0:
				av_num_zeros_per_image = X2.data.nonzero().numpy().shape[0]/y_dims[0]
				percent_zeros_per_image = 100*av_num_zeros_per_image/(y_dims[2]*y_dims[3])
				l2_error = np.sum((Y-self.reverse(X2)).data.numpy()**2)
				l1_error = np.sum(np.abs(X2.data.numpy()))
				fista_error = l2_error + self.tau*l1_error
				print("Iteration: "+repr(i+1) + ", l2 error:" + repr(l2_error) + ", l1 error: "+repr(l1_error)+ ", Total FISTA error: {0:1.2f}".format(fista_error) + ", Av. sparsity: {0:1.2f}".format(percent_zeros_per_image) +"%")
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




class SL_CSC_IHT(nn.Module):
	def __init__(self, stride=1, dp_channels=1, atom_r=1, atom_c=1, numb_atom=1, T_SC=1, k=1):
		super(SL_CSC_IHT, self).__init__()
		self.D_trans = nn.Conv2d(dp_channels, numb_atom, (atom_r, atom_c), stride, padding=0, dilation=1, groups=1, bias=False)
		self.D = nn.ConvTranspose2d(numb_atom, dp_channels, (atom_c, atom_r), stride, padding=0, output_padding=0, groups=1, bias=False, dilation=1)
		self.D_trans.weight.data = self.D.weight.data.permute(0,1,3,2)
		self.k = k
		self.T_SC=T_SC
		self.forward_type = 'IHT'

	def forward(self, Y):
		print("Running IHT")
		y_dims = list(Y.data.size())
		w_dims = list(self.D_trans.weight.data.size())
		# Randomly initialise X
		X = Variable(torch.randn(y_dims[0], w_dims[0], (y_dims[2]-w_dims[2]+1),(y_dims[3]-w_dims[3]+1)))
		HT_arg = X - self.D_trans(self.D(X)-Y)
		for i in range(0, self.T_SC):
			# Hard threshold each image in the dataset
			X = keep_k_largest(HT_arg, self.k)
			# Update HT arg as long as is not the last iteration
			if i < self.T_SC:
				HT_arg = X - self.D_trans(self.D(X)-Y)

			if (i+1)%5== 0:
				# After run IHT print out the result
				av_num_zeros_per_image = X.data.nonzero().numpy().shape[0]/y_dims[0]
				percent_zeros_per_image = 100*av_num_zeros_per_image/(y_dims[2]*y_dims[3])
				l2_error = np.sum((Y-self.reverse(X)).data.numpy()**2)
				print("After " +repr(i+1) + " iterations of IHT, l2 error:" + repr(l2_error) + " , Av. sparsity: {0:1.2f}".format(percent_zeros_per_image) +"%")
		return X		

		

	def reverse(self, x):
		out = self.D(x)
		return out
		




		