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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
	CSC = SL_CSC_IHT_backtracking(loaded_CSC_vars["stride"], loaded_CSC_vars["dp_channels"], loaded_CSC_vars["atom_r"], loaded_CSC_vars["atom_c"], loaded_CSC_vars["numb_atom"], loaded_CSC_vars["T_SC"], loaded_CSC_vars["k"])
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
	CSC = SL_CSC_FISTA_backtracking(loaded_CSC_vars["stride"], loaded_CSC_vars["dp_channels"], loaded_CSC_vars["atom_r"], loaded_CSC_vars["atom_c"], loaded_CSC_vars["numb_atom"], loaded_CSC_vars["tau"], loaded_CSC_vars["T_SC"], loaded_CSC_vars["T_PM"])
	# Load in network parameters
	CSC.load_state_dict(torch.load(torch_load_path))
	# Return model 
	return CSC

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
	# x is a pytorch variable, extract data tensor and convert to a numpy array
	X_numpy = X.data.numpy()
	X_dims = list(np.shape(X_numpy))
	X_new = np.zeros((X_dims[0], X_dims[1], X_dims[2], X_dims[3]))	
	for i in range(X_dims[0]):
		# copy image data, want to sort elements only for a given image
		abs_coeffs = np.absolute(np.copy(X_numpy[i]))
		# extract a list of the ordered elements
		inds_ordered = np.dstack(np.unravel_index(np.argsort(abs(abs_coeffs).ravel()), (X_dims[1], X_dims[2], X_dims[3])))
		# Identify the support of the k largest elements
		sup = inds_ordered[0][-k:]
		# Update X_new all but the k largest entries of x
		for j in range(len(sup)):
			X_new[i][sup[j][0]][sup[j][1]][sup[j][2]] = X_numpy[i][sup[j][0]][sup[j][1]][sup[j][2]]
	X_out = Variable(torch.from_numpy(X_new).type(torch.FloatTensor))
	return X_out, sup

def project_onto_sup(X, sup):
	X_numpy = X.data.numpy()
	X_dims = list(np.shape(X_numpy))
	X_new = np.zeros((X_dims[0], X_dims[1], X_dims[2], X_dims[3]))
	for i in range(X_dims[0]):
		for j in range(len(supp)):
			X_new[i][sup[j][0]][sup[j][1]][sup[j][2]] = X_numpy[i][sup[j][0]][sup[j][1]][sup[j][2]]
	X_out = Variable(torch.from_numpy(X_new).type(torch.FloatTensor))
	return X_out

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
			# Fix dictionary and calculate sparse code
			if CSC.forward_type == 'FISTA_fixed_step':
				CSC.calc_L(input_dims)
			if i < 0:
				X = CSC.D_trans(inputs).detach()
			else:
				X = CSC.forward(inputs).detach()
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
				if j==0 or (j+1)%20 == 0:
					print("Average loss per data point at iteration {0:1.0f}".format(j+1) + " of SGD: {0:1.4f}".format(np.asscalar(loss.data.numpy())))
					plt.figure(1)
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
			
			l2_error_percent = 100*np.sum((inputs-CSC.D(X)).data.numpy()**2)/ np.sum((inputs).data.numpy()**2)
			print("After " +repr(j+1) + " iterations of SGD, average l2 error over batch: {0:1.2f}".format(l2_error_percent) + "%")
			# Normalise each atom / kernel
			CSC.normalise_weights()
			# Ensure that weights for the reverse and forward operations are consistent	
			CSC.D_trans.weight.data = CSC.D.weight.data.permute(0,1,3,2)
	# Return trained CSC
	return CSC

		
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# CSC CLASSES AND CONSISTENCY FUNCTIONS
class SL_CSC_FISTA(nn.Module):
	def __init__(self, stride=1, dp_channels=1, atom_r=1, atom_c=1, numb_atom=1, tau=1, T_SC=1, T_PM=1):
		super(SL_CSC_FISTA, self).__init__()
		self.D_trans = nn.Conv2d(dp_channels, numb_atom, (atom_r, atom_c), stride, padding=0, dilation=1, groups=1, bias=False)
		self.D = nn.ConvTranspose2d(numb_atom, dp_channels, (atom_c, atom_r), stride, padding=0, output_padding=0, groups=1, bias=False, dilation=1)
		# self.dropout = nn.Dropout2d(p=0.5, inplace=False)
		self.normalise_weights()
		self.D_trans.weight.data = self.D.weight.data.permute(0,1,3,2)
		self.tau = tau
		self.T_SC = T_SC
		self.T_PM = T_PM
		self.forward_type = 'FISTA'

	def forward(self, Y):
		print("Running FISTA to recover/ estimate sparse code")
		# Initialise t variables needed for FISTA
		t1 = 1
		# Initialise X1 Variable - note we need X1 and X2 as we need to use the prior two prior estimates for each update
		y_dims = list(Y.data.size())
		w_dims = list(self.D_trans.weight.data.size())
		# Initialise our guess for X
		X1 = Variable(torch.rand(y_dims[0], w_dims[0], (y_dims[2]-w_dims[2]+1),(y_dims[3]-w_dims[3]+1)))
		# Calculate first update
		X2, FISTA_error, alpha = self.linesearch(Y,X1)
		# Iterate FISTA for a prescribed number of iterations
		print("Number of iterations running FISTA for: " + repr(self.T_SC))
		for i in range(0, self.T_SC):
			# Update t variables
			t2 = (1 + np.sqrt(1+4*(t1**2)))/2 
			# Update Z
			Z = X2 + (X2-X1)*(t1 - 1)/t2
			# Construct minimizer argument to feed into the soft thresholding function
			# X1 = X2.clone() #temp
			X2, FISTA_error, alpha = self.linesearch(Y,Z)
			# Update variables for next iteration
			X1 = X2.clone() #untoggle
			t1 = t2
			# Print at intervals to present progress
			if i==0 or (i+1)%10 == 0:
				av_num_zeros_per_image = X2.data.nonzero().numpy().shape[0]/y_dims[0]
				percent_zeros_per_image = 100*av_num_zeros_per_image/(y_dims[2]*y_dims[3])
				l2_error = np.sum((Y-self.D(X2)).data.numpy()**2)
				l1_error = np.sum(np.abs(X2.data.numpy()))
				# pix_error = l2_error/(y_dims[0]*y_dims[2]*y_dims[3])
				error_percent = l2_error*100/(np.sum((Y).data.numpy()**2))
				# print("Iteration: "+repr(i) + ", l2 error:{0:1.2f}".format(l2_error) + ", l1 error: {0:1.2f}".format(l1_error) + ", l2 error percent: {0:1.2f}".format(error_percent)+ "%, Total FISTA error: {0:1.2f}".format(FISTA_error) + ", Av. sparsity: {0:1.2f}".format(percent_zeros_per_image) +"%")
				print("After " +repr(i+1) + " iterations of FISTA, average l2 error over batch: {0:1.2f}".format(error_percent) + "% , Av. sparsity per image: {0:1.2f}".format(percent_zeros_per_image) +"%")
		return X2


	def reverse(self, X):
		out = self.D(X)
		return out

	def linesearch(self,Y,X):
		# Define search parameter for Armijo method
		c = 0.5
		alpha = 1
		g = self.D_trans(Y-self.D(X))
		ST_arg = X + alpha*g
		X_update = soft_thresh(ST_arg, self.tau*alpha)
		# Calculate cost of current X location
		l1_error = np.sum(np.abs(X.data.numpy()))
		l2_error = np.sum((Y-self.D(X)).data.numpy()**2)
		current_cost = l2_error + self.tau*l1_error
		# print("Cost at the beginning of the linesearch: {0:1.2f}".format(current_cost)+", l2 error:{0:1.2f}".format(l2_error) + ", l1 error: {0:1.2f}".format(l1_error))
		# Calculate the cost of the updated position
		update_cost = np.sum((Y-self.D(X_update)).data.numpy()**2) + self.tau*np.sum(np.abs(X.data.numpy()))
		# While the cost at the next location is higher than the current one iterate
		count = 0
		while update_cost >= current_cost and count<=15:
			alpha = alpha*c
			ST_arg = X + alpha*g
			X_update = soft_thresh(ST_arg, self.tau*alpha)
			l1_error = np.sum(np.abs(X_update.data.numpy()))
			l2_error = np.sum((Y-self.D(X_update)).data.numpy()**2)
			update_cost = l2_error + self.tau*l1_error
			count +=1
		# print("Cost at the end of the linesearch: {0:1.2f}".format(update_cost)+ ", l2 error:{0:1.2f}".format(l2_error) + ", l1 error: {0:1.2f}".format(l1_error))
		return X_update, update_cost, alpha

	def normalise_weights(self):
		print("Normalising kernels")
		filter_dims = list(np.shape(self.D.weight.data.numpy()))
		for i in range(filter_dims[0]):
			for j in range(filter_dims[1]):
				l2_norm = ((np.sum(self.D.weight.data[i][j].numpy()**2))**0.5)
				if l2_norm > 10^(-7): 
					self.D.weight.data[i][j] = self.D.weight.data[i][j]/l2_norm
				else:
					print("Kernel with 0 l2 norm identified, setting to zero")
					self.D.weight.data[i][j] = torch.zeros(filter_dims[2], filter_dims[3])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SL_CSC_IHT(nn.Module):
	def __init__(self, stride=1, dp_channels=1, atom_r=1, atom_c=1, numb_atom=1, T_SC=1, k=1):
		super(SL_CSC_IHT, self).__init__()
		self.D_trans = nn.Conv2d(dp_channels, numb_atom, (atom_r, atom_c), stride, padding=0, dilation=1, groups=1, bias=False)
		# self.dropout = nn.Dropout2d(p=0.5, inplace=False)
		self.D = nn.ConvTranspose2d(numb_atom, dp_channels, (atom_c, atom_r), stride, padding=0, output_padding=0, groups=1, bias=False, dilation=1)
		self.normalise_weights()
		self.D_trans.weight.data = self.D.weight.data.permute(0,1,3,2)
		self.k = k
		self.T_SC=T_SC
		self.forward_type = 'IHT'

	def forward(self, Y):
		print("Running IHT")
		y_dims = list(Y.data.size())
		w_dims = list(self.D_trans.weight.data.size())
		# Initialise X as zerio tensor
		X = Variable(torch.zeros(y_dims[0], w_dims[0], (y_dims[2]-w_dims[2]+1),(y_dims[3]-w_dims[3]+1)))
		# X = self.D_trans(Y)
		for i in range(0, self.T_SC):
			# print(np.sum(X[0].data.numpy()**2))
			# Hard threshold each image in the dataset
			X, l2_error, alpha = self.linesearch(Y,X)
			if i==0 or (i+1)%10 == 0:
				# After run IHT print out the result
				av_num_zeros_per_image = X.data.nonzero().numpy().shape[0]/y_dims[0]
				percent_zeros_per_image = 100*av_num_zeros_per_image/(y_dims[2]*y_dims[3])
				# pix_error = l2_error/(y_dims[0]*y_dims[2]*y_dims[3])
				error_percent = l2_error*100/(np.sum((Y).data.numpy()**2))
				print("After " +repr(i) + " iterations of IHT, average l2 error over batch: {0:1.2f}".format(error_percent) + "% , Av. sparsity per image: {0:1.2f}".format(percent_zeros_per_image) +"%")
		return X

	
	def reverse(self, x):
		out = self.D(x)
		return out

	def normalise_weights(self):
		filter_dims = list(np.shape(self.D.weight.data.numpy()))
		for i in range(filter_dims[0]):
			for j in range(filter_dims[1]):
				self.D.weight.data[i][j] = self.D.weight.data[i][j]/((np.sum(self.D.weight.data[i][j].numpy()**2))**0.5)


	def linesearch(self,Y,X):
		# Define search parameter for Armijo method
		c = 0.5
		alpha = 1
		g = self.D_trans(Y-self.D(X))
		HT_arg = X + alpha*g
		X_update, sup = hard_threshold_k(HT_arg, self.k)
		# Calculate cost of current X location
		l2_error_start = np.sum((Y-self.D(X)).data.numpy()**2)
		# Calculate cost of first suggested update
		l2_error = np.sum((Y-self.D(X_update)).data.numpy()**2)
		# While the cost at the next location is higher than the current one iterate up to a count of 8
		count = 0
		while l2_error >= l2_error_start and count<=15:
			alpha = alpha*c
			HT_arg = X + alpha*g
			X_update, sup = hard_threshold_k(HT_arg, self.k)
			l2_error = np.sum((Y-self.D(X_update)).data.numpy()**2)
			count +=1
		# print("l2 error at end of linesearch step:{0:1.2f}".format(l2_error))
		return X_update, l2_error, alpha
			

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SL_CSC_OMP(nn.Module):
	def __init__(self, stride=1, dp_channels=1, atom_r=1, atom_c=1, numb_atom=1, T_SC=1, k=1):
		super(SL_CSC_OMP, self).__init__()
		self.D_trans = nn.Conv2d(dp_channels, numb_atom, (atom_r, atom_c), stride, padding=0, dilation=1, groups=1, bias=False)
		# self.dropout = nn.Dropout2d(p=0.5, inplace=False)
		self.D = nn.ConvTranspose2d(numb_atom, dp_channels, (atom_c, atom_r), stride, padding=0, output_padding=0, groups=1, bias=False, dilation=1)
		self.normalise_weights()
		self.D_trans.weight.data = self.D.weight.data.permute(0,1,3,2)
		self.k = k
		self.T_SC=T_SC
		self.forward_type = 'OMP'
		self.stride = stride

	def forward(self, Y):
		print("Running OMP")
		y_dims = list(Y.data.size())
		w_dims = list(self.D_trans.weight.data.size())
		# Initialise X as zerio tensor
		X = Variable(torch.zeros(y_dims[0], w_dims[0], (y_dims[2]-w_dims[2]+1),(y_dims[3]-w_dims[3]+1)))
		R = Y.clone()
		# Initialise numy arrays to store indices storing the kernels in the support
		kernel_inds = np.zeros((y_dims[0], self.k))
		row_inds = np.zeros((y_dims[0], self.k))
		column_inds = np.zeros((y_dims[0], self.k))
		# Define optimisation procedure to recover X
		learning_rate= 1
		momentum = 0.9
		weight_decay = 0
		cost_function = nn.MSELoss(size_average=True)
		# Calculate Y l2 norm
		Y_l2 = np.sum(R.data.numpy()**2)

		# To observe residual as it proceeds
		plt.ion()
		plt.show()

		# Iteratively add elements to the support
		for i in range(1, self.k):
			# Calculate the atoms of D that correlate most highly with the residual of each element
			Z = self.D_trans(R).data.numpy()
			# print(Z)
			# Iterate through each data point to work out the individual supports
			for j in range(y_dims[0]):
				# Identify new kernel to add to support
				# Note that our version of OMP is slightly different.. if we use filter sizes less than the dimension of the signal then we loose the location of the filter
				# In other words instead of adding the filter at one location, we add it at all.
				kernel_inds[j, i-1], row_inds[j,i-1], column_inds[j,i-1] = np.unravel_index(Z[j].argmax(), Z[j].shape)
				print("Kernels selected:")
				print(kernel_inds[j,:])
				# Extract the kernels in the support and move into a new convolutional filter
				supp_D = nn.ConvTranspose2d(i, y_dims[1], (w_dims[3], w_dims[2]), self.stride, padding=0, output_padding=0, groups=1, bias=False, dilation=1)
				# Iteratively load the support into convolution operator
				for l in range(i):
					supp_D.weight.data[l] = self.D.weight.data[kernel_inds[j,l]]
				# Define a temporary variable for X[j] to update and iterate over with the optimiser
				temp_X_j = Variable(torch.zeros(1, i, (y_dims[2]-w_dims[2]+1), (y_dims[3]-w_dims[3]+1)), requires_grad = True)
				# Define optimisation procedure
				optimizer2 = torch.optim.SGD([temp_X_j], lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)
				# Run a number of iteration steps to minimise the l2 error ||Ax-y||
				for k in range(self.T_SC):
					# Zero the gradient
					optimizer2.zero_grad()
					# Calculate estimate of reconstructed Y
					Y_j_recon = supp_D(temp_X_j)
					# Calculate loss according to the defined cost function between the true Y and reconstructed Y
					loss = cost_function(Y_j_recon, Y[j])
					# Calculate the gradient of the cost function wrt to each parameters
					loss.backward()
					print(loss)
					# print(temp_X_j.grad)
					# Update each parameter according to the optimizer update rule (single step)
					optimizer2.step()
				# Update the sparse representation of X
				for l in range(i):
					X.data[j][kernel_inds[j, l]] = temp_X_j.data[0][l]

				Y_recon = supp_D(temp_X_j)
			#WIP! problem at this point
			# print(X)
			R1 = Y - self.D(X)
			R2 = Y - Y_recon

			print("Difference between two methods for calculating R")
			print(R1-R2)


			plt.figure(2)
			plt.subplot(1,2,1)
			plt.imshow(Y.data[0][0].numpy(), cmap='gray')
			plt.title("Original Image")
			plt.subplot(1,2,2)
			plt.imshow(R2.data[0][0].numpy(), cmap='gray')
			plt.title("Residual of Image")
			plt.draw()
			plt.pause(0.001)			
			input("Press Enter to continue...")

			if (i)%1 == 0:
				R_l2_error = np.sum(R.data.numpy()**2)
				R_l2_percent_error = 100*R_l2_error/Y_l2
				print("l2 norm of residual at cardinality 0: {0:1.2f}%".format(R_l2_percent_error))
		return X



	def reverse(self, x):
		out = self.D(x)
		return out

	def normalise_weights(self):
		filter_dims = list(np.shape(self.D.weight.data.numpy()))
		for i in range(filter_dims[0]):
			for j in range(filter_dims[1]):
				self.D.weight.data[i][j] = self.D.weight.data[i][j]/((np.sum(self.D.weight.data[i][j].numpy()**2))**0.5)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# class SL_CSC_NIHT(nn.Module):
# 	def __init__(self, stride=1, dp_channels=1, atom_r=1, atom_c=1, numb_atom=1, T_SC=1, k=1):
# 		super(SL_CSC_NIHT, self).__init__()
# 		self.D_trans = nn.Conv2d(dp_channels, numb_atom, (atom_r, atom_c), stride, padding=0, dilation=1, groups=1, bias=False)
# 		self.D = nn.ConvTranspose2d(numb_atom, dp_channels, (atom_c, atom_r), stride, padding=0, output_padding=0, groups=1, bias=False, dilation=1)
# 		self.normalise_weights()
# 		self.D_trans.weight.data = self.D.weight.data.permute(0,1,3,2)
# 		self.k = k
# 		self.T_SC=T_SC
# 		self.forward_type = 'IHT'

# 	def forward(self, Y):
# 		print("Running NIHT")
# 		y_dims = list(Y.data.size())
# 		w_dims = list(self.D_trans.weight.data.size())
# 		# Initialise x as variable containing zero tensor
# 		X = Variable(torch.zeros(y_dims[0], w_dims[0], (y_dims[2]-w_dims[2]+1),(y_dims[3]-w_dims[3]+1)))
# 		temp, sup = hard_threshold_k(self.D_trans(Y), self.k)
# 		for i in range(self.T_SC):
# 			g = self.D_trans(y - self.D(X))
# 			g_supp = 
# 			mu = (np.sum(g.data.numpy()**2))/(np.sum(self.D(g)))
# 			# WIP

# 	def reverse(self, x):
# 		out = self.D(x)
# 		return out

# 	def normalise_weights(self):
# 		filter_dims = list(np.shape(self.D.weight.data.numpy()))
# 		for i in range(filter_dims[0]):
# 			for j in range(filter_dims[1]):
# 				self.D.weight.data[i][j] = self.D.weight.data[i][j]/((np.sum(self.D.weight.data[i][j].numpy()**2))**0.5)	




		