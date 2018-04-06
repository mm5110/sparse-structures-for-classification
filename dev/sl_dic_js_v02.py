import numpy as np
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
	z = np.abs(x) - alpha
	z[z<0] = 0
	y = np.multiply(z, np.sign(x))
	return y

def conv_mat_mult():
	pass

def trans_conv_mat_mult():
	pass

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DICTIONARY LEARNING FUNCTIONS
def FISTA(y, A, alpha, T, ):
	t1 = 0
	t2 = (1 + np.sqrt(1+4*t1^2))/2
	x1 = np.zeros()
	minimizer_arg = (2/L)*trans_conv_mat_mult(A,y)

	for i in xrange(1,T):
		# Calculate latest sparse code estimate
		x2 = soft_thresh(minimizer_arg)

		# If this was not the last iteration then update variables for the next iteration
		if i <T:
			# Update t variables
			t1 = t2
			t2 = (1 + np.sqrt(1+4*t2^2))/2

			# Update z
			z = x + (x2-x1)*(t1 - 1)/t2

			# Construct minimizer argument to feed into the soft thresholding function
			minimizer_arg = z-(2/L)*trans_conv_mat_mult(A,(conv_mat_mult(A,z)-y))

			# Update x1 to calculate next z value
			x1 = x2

	return x2

# def FISTA_w_backtracking(y, A, alpha):

# def CGIHT(y, A, Lambda):

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# WRAPPER FUNCTIONS
# def update_dictionaries():

# def recover_sparse_rep():
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def train_SL_CSC(Y, T, stride, c, kdr, kdc, numb_k):
	# Initialise convolutional dictionary:.t
	D = nn.Conv2d(c, numb_k, (kdr,kdc), stride, padding=0, dilation=1, groups=1, bias=False)
	return D


# MAIN LOOP TESTING DICTIONARY LEARNING ON SYNTHETIC DATA
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Synthetic training data dimensions:
d = 10
c = 3
dr = 28
dc = 28

# Local dictionary dimensions
kdr = 5
kdc = 5
numb_k = 3*kdr*kdc

# # Vectorised problem parameters - lets see if can do all in 2d space
# M = dat_rows*dat_cols
# m = loc_dic_rows*loc_dic_cols

# Generate synthetic training data to train model
Y = Variable(torch.randn(d, c, dr, dc).type(dtype))

# Set dictionary learning parameters
# Number of iterations
T = 100
stride = 1

D = train_SL_CSC(Y, T, stride, c, kdr, kdc, numb_k)
print(D.weight)

X = D(Y)
print(X)










