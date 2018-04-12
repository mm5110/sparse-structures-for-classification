import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
from torch.autograd import Variable


# FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TESTS:

# TEST softthresholding algorithm
# Shrinkage/soft threshold operator: COMMENT when satisfied
temp = Variable(torch.randn(2,1,3,3)*2)
temp_s = soft_thresh(temp, 2)
print("TEST: soft thresholding function")
print("Input argument: ")
print(temp)
print("Soft thresholding output: ")
print(temp_s)

# TEST FISTA
# Generate dictionary
A = np.random.randn(10,20)
A_gram = np.matmul(np.transpose(A),A)
# Calculate eigenvalues
eigen_vals, eigen_vecs = LA.eig(A_gram)
# Since D^TD is a symmetric matrix we know that the eigenvalues must be real
eigen_vals = np.real(eigen_vals)
# We take the largest positive eigenvalue as out value for L
L = 2*np.asscalar(np.amax(eigen_vals, axis=None))
print(L)

# Generate sparse signal
x = np.zeros((20,1))
locs = random.sample(range(20), 3)
signs = np.sign(np.random.randn(3,1))
coeffs = (np.random.rand(3,1)+1)*3
x[locs] = np.multiply(signs, coeffs)
y = np.matmul(A,x)


# Run FISTA algorithm on y and A
T = 1000
t1 = 0
t2 = (1 + np.sqrt(1+4*(t1**2)))/2
# Initialise X1 Variable - note we need X1 and X2 as we need to use the prior two prior estimates for each update
x1 = np.matmul(np.transpose(A),y)
# Minimizer argument
minimizer_arg = x1 - (2/L)*np.matmul(np.transpose(A),(np.matmul(A,x1)-y))

for i in range(0,T):
	# Calculate latest sparse code estimate
	r = np.absolute(minimizer_arg) - (2/L)
	r[r<0] = 0
	x2 = np.multiply(r, np.sign(x1))
	# print("Iteration: "+repr(i+1)+ ", Sparsity level: " +repr(X2.data.nonzero().numpy().shape[0]))

	# If this was not the last iteration then update variables for the next iteration
	if i <T:
		# Update t variables
		t1 = t2
		t2 = (1 + np.sqrt(1+4*(t1**2)))/2 

		# Update z
		z = x2 + (x2-x1)*(t1 - 1)/t2

		# Construct minimizer argument to feed into the soft thresholding function
		minimizer_arg = z - (2/L)*np.matmul(np.transpose(A),(np.matmul(A,z)-y))

		# Update x1 to calculate next z value
		x1 = x2

print(x)
print(x2

# Test  