import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

import random
import os
import yaml
import csv
import datetime

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
	CSC = SL_CSC_IHT(loaded_CSC_vars["stride"], loaded_CSC_vars["dp_channels"], loaded_CSC_vars["atom_r"], loaded_CSC_vars["atom_c"], loaded_CSC_vars["numb_atom"], loaded_CSC_vars["T_SC"], loaded_CSC_vars["k"])
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

def log_training_data(log_file, initialise, log_data, fieldnames):
	with open(log_file, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		if initialise ==True:
			writer.writeheader()
		writer = writer.writerow({fieldnames[0]: log_data[0], fieldnames[1]: log_data[1], fieldnames[2]: log_data[2], fieldnames[3]: log_data[3], fieldnames[4]: log_data[4], fieldnames[5]: log_data[5], fieldnames[6]: log_data[6], fieldnames[7]: log_data[7]})

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
	optimizer = torch.optim.SGD(CSC_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)
	# Initialise variables needed to plot a random sample of three kernels as they are trained
	filter_dims = list(np.shape(CSC.D_trans.weight.data.numpy()))
	idx = random.sample(range(0, filter_dims[0]), 3)
	# Prepare logging files and data
	time_str = str(datetime.datetime.now())
	fieldnames = ['Epoch', 'Batch Number', 'Total Batch Number', 'l2 SC', 'Number SC Iterations', 'l2 End', 'Average Sparsity', 'Cumulative Filters Trained']
	initialise = True
	training_log_filename = 'log_data/' + time_str + '_' + str(CSC.forward_type) + '_' + 'training_log.csv'
	activation_data_filename = 'log_data/' +time_str + '_' + str(CSC.forward_type) + '_' + 'activations'
	filter_activations = np.zeros(filter_dims[0])	
	# Prepare plots of filters
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
			CSC.batch_size = input_dims[0]
			# Generate dropout filter
			active_filter_inds = sample_filters(filter_dims[0], p, CSC.k)
			CSC.mask = create_dropout_mask(input_dims[0], filter_dims[0], (input_dims[2]-filter_dims[2]+1), (input_dims[3]-filter_dims[3]+1), active_filter_inds)
			# Fix dictionary and calculate sparse code
			if CSC.forward_type == 'FISTA_fixed_step':
				CSC.calc_L(input_dims)
			# if i < 3:
			# 	X = CSC.D_trans(inputs).detach()
			# else:
			# 	X = CSC.forward(inputs).detach()
			X, SC_error_percent, numb_SC_iterations, filters_selected = CSC.forward(inputs)
			X = X.detach()
			# filters_selected = filters_selected.numpy()
			for l in range(len(filters_selected)):
				filter_activations[filters_selected[l][1]] = filter_activations[filters_selected[l][1]] + 1

			plt.figure(10)
			plt.clf()
			plt.hist(filter_activations, bins=50)  # arguments are passed to np.histogram
			plt.title("Histogram of filter activations, epoch Number: {0:1.0f}".format(epoch) + ", batch number: {0:1.0f}".format(i+1))
			plt.ylabel("Number of filters")
			plt.xlabel("Activation frequency bin")
			plt.draw()
			plt.pause(0.001)

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
				if j==0 or (j+1)%20 == 0:
					print("Average loss per data point at iteration {0:1.0f}".format(j+1) + " of SGD: {0:1.4f}".format(np.asscalar(loss.data.numpy())))
					plt.figure(1)
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

			l2_error_percent = 100*np.sum((inputs-CSC.D(X)).data.numpy()**2)/ np.sum((inputs).data.numpy()**2)
			print("After " +repr(j+1) + " iterations of SGD, average l2 error over batch: {0:1.2f}".format(l2_error_percent) + "%")
			# Normalise each atom / kernel
			CSC.normalise_weights()
			# Ensure that weights for the reverse and forward operations are consistent	
			CSC.D_trans.weight.data = CSC.D.weight.data.permute(0,1,3,2)
			# Reset optimizer
			optimizer = torch.optim.SGD(CSC_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)
			# Log training data
			log_data = [epoch, i, epoch*batch_size+i, SC_error_percent, numb_SC_iterations, l2_error_percent, average_number_nonzeros, np.count_nonzero(filter_activations)]
			log_training_data(training_log_filename, initialise, log_data, fieldnames)
			initialise = False
	# Reset the mask for non training state (i.e. no dropout)
	CSC.mask = torch.ones(input_dims[0], filter_dims[0], (input_dims[2]-filter_dims[2]+1), (input_dims[3]-filter_dims[3]+1))
	# Save down the filter activations
	np.save(activation_data_filename, filter_activations)
	# Plot filter activations
	# print("Plotting filter activations")
	# plt.figure(9)
	# idx = np.arange(batch_size)
	# plt.bar(idx, filter_activations)
	# plt.show(block = True)
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
			if i==0 or (i+1)%2 == 0:
				av_num_zeros_per_image = X2.data.nonzero().numpy().shape[0]/y_dims[0]
				percent_zeros_per_image = 100*av_num_zeros_per_image/(y_dims[2]*y_dims[3])
				l2_error = np.sum((Y-self.D(X2)).data.numpy()**2)
				l1_error = np.sum(np.abs(X2.data.numpy()))
				# pix_error = l2_error/(y_dims[0]*y_dims[2]*y_dims[3])
				error_percent = l2_error*100/(np.sum((Y).data.numpy()**2))
				# print("Iteration: "+repr(i) + ", l2 error:{0:1.2f}".format(l2_error) + ", l1 error: {0:1.2f}".format(l1_error) + ", l2 error percent: {0:1.2f}".format(error_percent)+ "%, Total FISTA error: {0:1.2f}".format(FISTA_error) + ", Av. sparsity: {0:1.2f}".format(percent_zeros_per_image) +"%")
				print("After " +repr(i+1) + " iterations of FISTA, average l2 error over batch: {0:1.2f}".format(error_percent) + "% , Av. sparsity per image: {0:1.2f}".format(percent_zeros_per_image) +"%")
		return X2, error_percent


	def reverse(self, X):
		out = self.D(X)
		return out

	def linesearch(self,Y,X):
		# Define search parameter for Armijo method
		c = 0.5
		alpha = 1
		alpha_lim = 15
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
		while update_cost >= current_cost and count<alpha_lim:
			alpha = alpha*c
			ST_arg = X + alpha*g
			X_update = soft_thresh(ST_arg, self.tau*alpha)
			l1_error = np.sum(np.abs(X_update.data.numpy()))
			l2_error = np.sum((Y-self.D(X_update)).data.numpy()**2)
			update_cost = l2_error + self.tau*l1_error
			count +=1
		if count >= alpha_lim:
			X_update = X
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
		# self.k_target = k
		# self.k=np.minimum(11*k, atom_r*atom_c)
		# self.k_lin_decay_rate = 10/10
		self.k = k
		self.T_SC=T_SC
		self.forward_type = 'IHT'
		self.batch_size = 1
		self.mask = torch.ones(self.batch_size, numb_atom, atom_r, atom_c)


	def forward(self, Y):
		print("Running IHT, projecting onto support cardinality k = {0:1.0f}".format(self.k))
		y_dims = list(Y.data.size())
		w_dims = list(self.D_trans.weight.data.size())
		# Initialise X as zero tensor
		X1 = Variable(torch.zeros(y_dims[0], w_dims[0], (y_dims[2]-w_dims[2]+1),(y_dims[3]-w_dims[3]+1)))
		alpha = 0.005 # Delete after testing
		X1_error = np.sum((Y).data.numpy()**2)
		X2_error = 0
		i=0
		run = True
		# for i in range(0, self.T_SC):
		while run == True:
			g = self.dropout(self.D_trans(Y-self.D(self.dropout(X1))))
			HT_arg = X1 + alpha*g
			X2, filters_selected = hard_threshold_k(HT_arg, self.k)
			X2_error = np.sum(((Y-self.D(self.dropout(X2))).data.numpy())**2)
			# print(X1_error)
			# print(X2_error)
			if X2_error < X1_error:
				X1 = X2
				X1_error = X2_error
			else:
				run = False
			# X, l2_error, alpha = self.linesearch(Y,X)# uncomment
			if i==0 or (i+1)%10 == 0:
				# After run IHT print out the result
				l2_error = X1_error
				av_num_zeros_per_image = X1.data.nonzero().numpy().shape[0]/y_dims[0]
				percent_zeros_per_image = 100*av_num_zeros_per_image/(y_dims[2]*y_dims[3])
				# pix_error = l2_error/(y_dims[0]*y_dims[2]*y_dims[3])
				error_percent = l2_error*100/(np.sum((Y).data.numpy()**2))
				print("After " +repr(i+1) + " iterations of IHT, average l2 error over batch: {0:1.2f}".format(error_percent) + "% , Av. sparsity per image: {0:1.2f}".format(percent_zeros_per_image) +"%")
			i=i+1
		# # Update k
		# if self.k > self.k_target:
		# 	temp = self.k - int(self.k_lin_decay_rate*self.k_target)
		# 	if temp <= self.k_target:
		# 		self.k = self.k_target
		# 	else:
		# 		self.k = temp
		# Return value of k calculated
		return X1, error_percent, i, filters_selected

	
	def reverse(self, x):
		out = self.D(x)
		return out

	def normalise_weights(self):
		filter_dims = list(np.shape(self.D.weight.data.numpy()))
		for i in range(filter_dims[0]):
			for j in range(filter_dims[1]):
				self.D.weight.data[i][j] = self.D.weight.data[i][j]/((np.sum(self.D.weight.data[i][j].numpy()**2))**0.5)

	def dropout(self,X):
		X_dropout = Variable(X.data*self.mask)
		return X_dropout

	# def linesearch(self,Y,X):
	# 	# Define search parameter for Armijo method
	# 	c = 0.5
	# 	alpha = 10
	# 	alpha_lim = 15
	# 	g = self.D_trans(Y-self.D(X))
	# 	HT_arg = X + alpha*g
	# 	X_update = hard_threshold_k(HT_arg, self.k)
	# 	# Calculate cost of current X location
	# 	l2_error_start = np.sum((Y-self.D(X)).data.numpy()**2)
	# 	# Calculate cost of first suggested update
	# 	l2_error = np.sum((Y-self.D(X_update)).data.numpy()**2)
	# 	# While the cost at the next location is higher than the current one iterate up to a count of 8
	# 	count = 0
	# 	while l2_error >= l2_error_start and count<alpha_lim:
	# 		alpha = alpha*c
	# 		HT_arg = X + alpha*g
	# 		X_update = hard_threshold_k(HT_arg, self.k)
	# 		l2_error = np.sum((Y-self.D(X_update)).data.numpy()**2)
	# 		count +=1
	# 	if count >= alpha_lim:
	# 		X_update = X
	# 	return X_update, l2_error, alpha
			

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




		