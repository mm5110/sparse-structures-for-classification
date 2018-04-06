import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

# COMPUTATION SETTINGS
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




# FUNCTIONS
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DATA FORMATTING FUNCTIONS
# def vectorise_var(Y):


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BASIC ALGORITHM FUNCTIONS
def soft_thresh(x, alpha):
	# INPUTS: x - is a numpy array, alpha - is the parameter of the soft thresholding function
	# OUTPUTS: y - is numpy array containing the elementwise soft thresholded values of x
	x_numpy = x.data.numpy() 
	z = np.absolute(x_numpy) - alpha
	print(type(z))
	z[z<0] = 0
	y_numpy = np.multiply(z, np.sign(x_numpy))
	y = Variable(torch.from_numpy(y_numpy))
	return y

def calc_Lipshitz_constant_wrapped_D(D, stride):
	# Extract weights from pytorch conv2d class instance D
	weights_numpy_array = np.squeeze(D.weight.data.numpy())
	w_dims = weights_numpy_array.shape
	# Initialise matrix form of local dictionary
	loc_dic_r = w_dims[1]*w_dims[2]
	loc_dic_c = w_dims[0]
	local_dic = np.zeros((loc_dic_r, loc_dic_c))
	# Generate matrix form of local dictionary
	for i in range(w_dims[0]):
		local_dic[:,i] = np.ndarray.flatten(weights_numpy_array[i,:,:])

	# Define stripe dictionary dimensions
	stripe_dic_r = loc_dic_r
	stripe_dic_c = (2*int(np.ceil(loc_dic_r/stride))-1)*loc_dic_c 
	stripe_dic = np.zeros((stripe_dic_r, stripe_dic_c))

	# Generate stripe dictionary stripe dictionary dimensions
	j=0
	for i in range(1, stripe_dic_r, stride):
		stripe_dic[:i,j:(j+loc_dic_c)] = local_dic[:i,:]
		j = j+loc_dic_c

	for i in range(0, stripe_dic_r, stride):
		stripe_dic[i:,j:(j+loc_dic_c)] = local_dic[:loc_dic_r-i,:]
		j = j+loc_dic_c
	
	# Visualise to check that have generated the matrix form of the stipe dictionary properly, comment when not needed
	# plt.imshow(stripe_dic, cmap='hot', interpolation='nearest')
	# plt.show()

	# Calculate the normal matrix of the stripe dictionary
	stripe_dic_norm = np.matmul(np.transpose(stripe_dic),stripe_dic)
	eigen_vals = LA.eig(stripe_dic_norm)
	print(eigen_vals)
	L = 2*np.amax(eigen_vals)
	return L

def calc_Lipshitz_constant(D, stride):
	# Extract weights from pytorch conv2d class instance D
	weights_numpy_array = np.squeeze(D.weight.data.numpy())
	w_dims = weights_numpy_array.shape
	# Initialise matrix form of local dictionary
	loc_dic_r = w_dims[1]*w_dims[2]
	loc_dic_c = w_dims[0]
	local_dic = np.zeros((loc_dic_r, loc_dic_c))
	# Generate matrix form of local dictionary
	for i in range(w_dims[0]):
		local_dic[:,i] = np.ndarray.flatten(weights_numpy_array[i,:,:])

	# Define stripe dictionary dimensions
	stripe_dic_r = loc_dic_r
	stripe_dic_c = int(np.ceil(loc_dic_r/stride))*loc_dic_c 
	stripe_dic = np.zeros((stripe_dic_r, stripe_dic_c))

	# Generate stripe dictionary stripe dictionary dimensions
	j=0

	for i in range(0, stripe_dic_r, stride):
		stripe_dic[i:,j:(j+loc_dic_c)] = local_dic[:loc_dic_r-i,:]
		j = j+loc_dic_c
	
	# Visualise to check that have generated the matrix form of the stipe dictionary properly, comment when not needed
	# plt.imshow(stripe_dic, cmap='hot', interpolation='nearest')
	# plt.show()

	# Calculate the normal matrix of the stripe dictionary
	stripe_dic_norm = np.matmul(np.transpose(stripe_dic),stripe_dic)
	eigen_vals = LA.eig(stripe_dic_norm)
	print(eigen_vals)
	L = 2*np.amax(eigen_vals)
	return L



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DICTIONARY LEARNING FUNCTIONS
def FISTA(Y, D, D_trans, T, L):
	# Initialise t variables needed for FISTA
	t1 = 0
	t2 = (1 + np.sqrt(1+4*(t1**2)))/2
	# Initialise X1 Variable - note we need X1 and X2 as we need to use the prior two prior estimates for each update
	X1 = D_trans(Y)
	# Minimizer argument
	minimizer_arg = X1 - (2/L)*D_trans((D(X1)-Y))

	for i in range(0,T):
		# Calculate latest sparse code estimate
		X2 = soft_thresh(minimizer_arg, (L/2))
		print("Iteration: "+repr(i+1)+ ", Sparsity level: " +repr(X2.data.nonzero().numpy().shape[0]))

		# If this was not the last iteration then update variables for the next iteration
		if i <T:
			# Update t variables
			t1 = t2
			t2 = (1 + np.sqrt(1+4*(t1**2)))/2 

			# Update z
			Z = X2 + (X2-X1)*(t1 - 1)/t2

			# Construct minimizer argument to feed into the soft thresholding function
			minimizer_arg = Z - (2/L)*D_trans((D(Z)-Y))

			# Update x1 to calculate next z value
			X1 = X2

	return X2

# def FISTA_w_backtracking(y, A, alpha):

# def CGIHT(y, A, Lambda):

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# WRAPPER FUNCTIONS
# def update_dictionaries():

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def train_SL_CSC(Y, T, T_FISTA, stride, dp_channels, atom_r, atom_c, numb_atom):
	# Initialise convolutional dictionary:
	# Define class with weight variable holding the transpose of the dictionary. D_trans(y) will execute D^Ty in vectorized form
	D_trans = nn.Conv2d(dp_channels, numb_atom, (atom_r, atom_c), stride, padding=0, dilation=1, groups=1, bias=False)
	# Define the dictionary layer as the transpose of the forward pass weight matrix. D(x) will execute Dx in vectorized form
	# D = nn.Conv2d(numb_atom, dp_channels, (atom_c, atom_r), stride, padding=0, dilation=1, groups=1, bias=False)
	D = nn.ConvTranspose2d(numb_atom, dp_channels, (atom_c, atom_r), stride, padding=0, output_padding=0, groups=1, bias=False, dilation=1)
	D.weight.data=D_trans.weight.data.permute(0,1,3,2)

	L = calc_Lipshitz_constant(D, stride)

	# Learn dictionary and sparse representation simultaneously
	for i in range(0,T):
		L = calc_Lipshitz_constant(D, stride)
		print(L)
		X = FISTA(Y, D, D_trans, T_FISTA, L)

	


# MAIN LOOP TESTING DICTIONARY LEARNING ON SYNTHETIC DATA
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Synthetic training data dimensions:
numb_dp = 5
dp_channels = 1
dp_r = 28
dp_c = 28

# Local dictionary dimensions
atom_r = 2
atom_c = 2
numb_atom = 3
# numb_atom = 1*atom_r*atom_c 


# Generate synthetic training data to train model
Y = Variable(torch.randn(numb_dp, dp_channels, dp_r, dp_c).type(dtype))

# Set dictionary learning parameters
# Number of iterations
T = 1
T_FISTA = 8
stride = 1

# TEST shrinkage/soft threshold operator: COMMENT when satisfied
# temp = Variable(torch.randn(2,1,3,3)*2)
# temp_s = soft_thresh(temp, 2)
# print("TEST: soft thresholding function")
# print("Input argument: ")
# print(temp)
# print("Soft thresholding output: ")
# print(temp_s)

# TEST FISTA


# Train SL-CSC model
D = train_SL_CSC(Y, T, T_FISTA, stride, dp_channels, atom_r , atom_c, numb_atom)










