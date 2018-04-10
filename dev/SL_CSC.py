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

	# Calculate the gram matrix of the stripe dictionary
	stripe_dic_gram = np.matmul(np.transpose(stripe_dic),stripe_dic)
	eigen_vals, eigen_vecs = LA.eig(stripe_dic_gram)
	# Since D^TD is a symmetric matrix we know that the eigenvalues must be real
	eigen_vals = np.real(eigen_vals)
	# We take the largest positive eigenvalue as out value for L
	L = 2*np.asscalar(np.amax(eigen_vals, axis=None))
	return L

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PRIMARY ALGORITHM FUNCTIONS
def FISTA(Y, CSC, T, L):
	# Initialise t variables needed for FISTA
	t1 = 0
	t2 = (1 + np.sqrt(1+4*(t1**2)))/2
	# Initialise X1 Variable - note we need X1 and X2 as we need to use the prior two prior estimates for each update
	X1 = CSC.forward(Y)
	# Minimizer argument
	ST_arg = X1 - (2/L)*CSC.forward((CSC.backward(X1)-Y))

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
			ST_arg = Z - (2/L)*CSC.forward((CSC.backward(Z)-Y))

			# Update X1 to calculate next Z value
			X1 = X2

	# Return final iteration of sparse code
	return X2


def train_SL_CSC(Y, CSC, T, T_FISTA, stride, dp_channels, atom_r, atom_c, numb_atom, cost_function, optimizer):
	# Train network
	for i in range(T):
		# Calculate sparse code
		L = calc_Lipshitz_constant(CSC.D, stride)
		X = FISTA(Y, CSC, T_FISTA, L)
		
		# Udate weight matrix

		
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CSC Class
class SL_CSC(nn.Module):
	def __init__(Y, T, T_FISTA, stride, dp_channels, atom_r, atom_c, numb_atom):
		super(Net, self).__init__()
		self.D_trans = nn.Conv2d(dp_channels, numb_atom, (atom_r, atom_c), stride, padding=0, dilation=1, groups=1, bias=False)
		self.D = nn.ConvTranspose2d(numb_atom, dp_channels, (atom_c, atom_r), stride, padding=0, output_padding=0, groups=1, bias=False, dilation=1)
		self.make_forward_backward_consistent()
    
	def forward(self, x):
		out = self.D_trans(x)
		return out

	def backward(self, x):
		 out = self.D(x)

	def make_forward_backward_consistent(self):
		D.weight.data=D_trans.weight.data.permute(0,1,3,2)
    

	

# MAIN LOOP
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Training Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001
T = 1
T_FISTA = 5
stride = 1
learning_rate = 0.001
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
CSC = SL_CSC(Y, T, T_FISTA, stride, dp_channels, atom_r , atom_c, numb_atom)

# Define training settings/ options
cost_function = nn.MSELoss()
optimizer = torch.optim.SGD(SSC.parameters(), lr=learning_rate, momentum=momentum)  
# optimizer = torch.optim.Adam(SSC.parameters(), lr=learning_rate)

# Create Convolutional Sparse Coder
CSC = train_SL_CSC(Y, CSC, T, T_FISTA, stride, dp_channels, atom_r, atom_c, numb_atom, cost_function, optimizer)













