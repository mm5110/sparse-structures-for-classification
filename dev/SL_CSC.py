import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# COMPUTATION SETTINGS
dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

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

def power_method(CSC, T_PM, Y):
	print("Calculating Lipschitz constant using power method with " + repr(T_PM) + " iterations")
	y_dims = list(Y.data.size())
	w_dims = list(CSC.D_trans.weight.data.size())
	# X = CSC.forward(Y)
	X = Variable(torch.rand(y_dims[0], w_dims[0], (y_dims[2]-w_dims[2]+1),(y_dims[3]-w_dims[3]+1))+2)
	for i in range(T_PM):
		X = CSC.D_gram(X)
		X = X/np.asscalar(np.sum((X*X).data.numpy()))
	rayleigh_q = np.asscalar(np.sum((CSC.D_gram(X)*X).data.numpy())/np.sum((X*X).data.numpy()))
	L = 2*rayleigh_q
	print("L value calculated: " + repr(L))
	return L


def calc_Lipshitz_constant(D, stride):
	print("Calculating Lipshitz constant by unpacking into matrix")
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

	# Calculate the gram matrix of the stripe dictionary
	stripe_dic_gram = np.matmul(np.transpose(stripe_dic),stripe_dic)
	eigen_vals, eigen_vecs = LA.eig(stripe_dic_gram)
	# Since D^TD is a symmetric matrix we know that the eigenvalues must be real
	eigen_vals = np.real(eigen_vals)
	# We take the largest positive eigenvalue as out value for L
	L = 2*np.asscalar(np.amax(np.abs(eigen_vals), axis=None))
	print("L value calculated: " + repr(L))
	return L

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PRIMARY ALGORITHM FUNCTIONS
def recover_sparse_code(Y, CSC, T, L, sparse_code_method):
	if sparse_code_method == 'FISTA':
		print("Running FISTA to recover/ estimate sparse code")
		# Initialise t variables needed for FISTA
		t1 = 0
		t2 = (1 + np.sqrt(1+4*(t1**2)))/2
		# Initialise X1 Variable - note we need X1 and X2 as we need to use the prior two prior estimates for each update
		X1 = CSC.forward(Y)
		
		# Minimizer argument
		ST_arg = X1 - (2/L)*CSC.forward(CSC.reconstruct(X1)-Y)

		for i in range(0,T):
			# Calculate latest sparse code estimate
			X2 = soft_thresh(ST_arg, (2/L))
			print("Iteration: "+repr(i+1)+ ", Sparsity level: " +repr(X2.data.nonzero().numpy().shape[0]))

			# If this was not the last iteration then update variables needed for the next iteration
			if i <T:
				# Update t variables
				t1 = t2
				t2 = (1 + np.sqrt(1+4*(t1**2)))/2 

				# Update Z
				Z = X2 + (X2-X1)*(t1 - 1)/t2

				# Construct minimizer argument to feed into the soft thresholding function
				ST_arg = Z - (2/L)*CSC.forward((CSC.reconstruct(Z)-Y))

				# Update X1 to calculate next Z value
				X1 = X2

	# Return sparse representation
	return X2

def update_dictionary(CSC, T_DIC, optimizer, cost_function, X, Y):
	print("Running dictionary update")
	# Update weight matrix
	for i in range(T_DIC):
		# Zero the gradient
		optimizer.zero_grad()
		# Calculate estimate of reconstructed Y
		Y_recon = CSC.reconstruct(X)
		# Calculate loss according to the defined cost function between the true Y and reconstructed Y
		loss = cost_function(Y_recon, Y)
		print("Average loss per data point at iteration " +repr(i) + " :" + repr(np.asscalar(loss.data.numpy())))
		# Calculate the gradient of the cost function wrt to each parameters
		loss.backward()
		print("D_trans gradients")
		print(CSC.D.weight.grad)
		# Update each parameter according to the optimizer update rule (single step)
		optimizer.step()
		print("D_trans values post update")
		print(CSC.D_trans.weight)
	CSC.make_forward_recon_consistent()


def train_SL_CSC(Y, CSC, T, T_SC, T_DIC, stride, sparse_code_method, cost_function, optimizer):
	# Train network
	for i in range(T):
		# Calculate Lipschitz constant of dictionary
		L = power_method(CSC, 10, Y)
		L2 = calc_Lipshitz_constant(CSC.D, stride)
		# Fix dictionary and calculate sparse code
		X = recover_sparse_code(Y, CSC, T_SC, L, sparse_code_method)
		# Fix sparse code and update dictionary
		update_dictionary(CSC, T_DIC, optimizer, cost_function, X, Y)
	return CSC

		
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CSC Class
class SL_CSC(nn.Module):
	def __init__(self, T, T_FISTA, stride, dp_channels, atom_r, atom_c, numb_atom):
		super(SL_CSC, self).__init__()
		self.D_trans = nn.Conv2d(dp_channels, numb_atom, (atom_r, atom_c), stride, padding=0, dilation=1, groups=1, bias=False)
		self.D = nn.ConvTranspose2d(numb_atom, dp_channels, (atom_c, atom_r), stride, padding=0, output_padding=0, groups=1, bias=False, dilation=1)
		self.make_forward_recon_consistent()
    
	def forward(self, x):
		out = self.D_trans(x)
		return out

	def reconstruct(self, x):
		 out = self.D(x)
		 return out

	def make_forward_recon_consistent(self):
		# self.D.weight.data=self.D_trans.weight.data.permute(0,1,3,2)
		self.D_trans.weight.data = self.D.weight.data.permute(0,1,3,2)

	def D_gram(self, x):
		out = self.D_trans(self.D(x))
		return out
    

	

# MAIN LOOP
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Training Parameters
# num_epochs = 5
# batch_size = 100
learning_rate = 0.001
T = 10
T_SC = 5
T_DIC = 5
stride = 1
learning_rate = 0.1
momentum = 0.9

# Local dictionary dimensions
atom_r = 2
atom_c = 2
numb_atom = 3

# Synthetic training data dimensions:
numb_dp = 5
dp_channels = 1
dp_r = 28
dp_c = 28

# Generate synthetic training data to train model, Y is the training data or input to the classifier
Y = Variable(torch.randn(numb_dp, dp_channels, dp_r, dp_c).type(dtype))

# Intitilise Convolutional Sparse Coder CSC
CSC = SL_CSC(T, T_SC, stride, dp_channels, atom_r, atom_c, numb_atom)

# Define training settings/ options
sparse_code_method = 'FISTA'
cost_function = nn.MSELoss()
optimizer = torch.optim.SGD(CSC.parameters(), lr=learning_rate, momentum=momentum)  
# optimizer = torch.optim.Adam(SSC.parameters(), lr=learning_rate)

# Train Convolutional Sparse Coder
CSC = train_SL_CSC(Y, CSC, T, T_SC, T_DIC, stride, sparse_code_method, cost_function, optimizer)












