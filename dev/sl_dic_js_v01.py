import numpy as np
import torch.nn as nn

# FUNCTIONS
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DATA FORMATTING FUNCTIONS

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


def trans_conv_mat_mult():

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





# MAIN LOOP TESTING DICTIONARY LEARNING ON SYNTHETIC DATA
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Synthetic input data parameters:
d = 10
c = 3
rows = 256
cols = 256

# Vectorised problem parameters
M = rows*cols
m = 

Y = Variable(torch.randn(10, 3, 256, 256))









