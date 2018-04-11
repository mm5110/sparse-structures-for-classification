import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torchvision
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

def power_method(CSC, CSC_inv, T_PM, Y):
	print("Calculating Lipschitz constant using power method with " + repr(T_PM) + " iterations")
	# Intialise initial and random guess of dominant singular vector
	y_dims = list(Y.data.size())
	w_dims = list(CSC.D_trans.weight.data.size())
	X = Variable(torch.randn(y_dims[0], w_dims[0], (y_dims[2]-w_dims[2]+1),(y_dims[3]-w_dims[3]+1))*3+4)
	# Normalise initialised vector in the l2 norm
	X = X/(np.asscalar(np.sum((X**2).data.numpy()))**0.5)
	# Perform T_PM iterations applying A^TA to x then normalising
	for i in range(T_PM):
		X = CSC.forward(CSC_inv.forward(X))
		X = X/(np.asscalar(np.sum((X**2).data.numpy()))**0.5)
	# Compute dominant singular value
	dominant_sv = (np.asscalar(np.sum((CSC.forward(CSC_inv.forward(X))**2).data.numpy())))**0.5
	L = 2*dominant_sv
	print("L value calculated: " + repr(L))
	return L

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PRIMARY ALGORITHM FUNCTIONS
def recover_sparse_code(Y, CSC, CSC_inv, T, L, sparse_code_method):
	if sparse_code_method == 'FISTA':
		print("Running FISTA to recover/ estimate sparse code")
		# Initialise t variables needed for FISTA
		t1 = 0
		t2 = (1 + np.sqrt(1+4*(t1**2)))/2
		# Initialise X1 Variable - note we need X1 and X2 as we need to use the prior two prior estimates for each update
		print(Y)
		X1 = CSC.forward(Y)
		print(X1)		
		# Minimizer argument
		ST_arg = X1 - (2/L)*CSC.forward(CSC_inv.forward(X1)-Y)
		for i in range(0,T):
			# Calculate latest sparse code estimate
			X2 = soft_thresh(ST_arg, (2/L))
			if i%100 == 0:
				print("Iteration: "+repr(i+1)+ ", Sparsity level: " +repr(X2.data.nonzero().numpy().shape[0]))
			# If this was not the last iteration then update variables needed for the next iteration
			if i <T:
				# Update t variables
				t1 = t2
				t2 = (1 + np.sqrt(1+4*(t1**2)))/2 
				# Update Z
				Z = X2 + (X2-X1)*(t1 - 1)/t2
				# Construct minimizer argument to feed into the soft thresholding function
				ST_arg = Z - (2/L)*CSC.forward((CSC_inv.forward(Z)-Y))
				# Update X1 to calculate next Z value
				X1 = X2
	# Return sparse representation
	return X2



def update_dictionary(CSC, CSC_inv, T_DIC, optimizer, cost_function, X, Y):
	print("Running dictionary update")
	# Update weight matrix
	for i in range(T_DIC):
		# Zero the gradient
		optimizer.zero_grad()
		# Calculate estimate of reconstructed Y
		Y_recon = CSC_inv.forward(X)
		# Calculate loss according to the defined cost function between the true Y and reconstructed Y
		loss = cost_function(Y_recon, Y)
		print("Average loss per data point at iteration " +repr(i) + " :" + repr(np.asscalar(loss.data.numpy())))
		# Calculate the gradient of the cost function wrt to each parameters
		loss.backward()
		# Update each parameter according to the optimizer update rule (single step)
		optimizer.step()
	make_consistent(CSC, CSC_inv)


def train_SL_CSC(train_loader, CSC, CSC_inv, num_epochs, T_SC, T_DIC, stride, sparse_code_method, cost_function, optimizer):
	for epoch in range(num_epochs):
		for i, (images, labels) in enumerate(train_loader):
			images = Variable(images)
			labels = Variable(labels)
			# Calculate Lipschitz constant of dictionary
			L = power_method(CSC, CSC_inv, 20, images)
			# Fix dictionary and calculate sparse code
			X = recover_sparse_code(images, CSC, CSC_inv, T_SC, L, sparse_code_method)
			# Fix sparse code and update dictionary
			update_dictionary(CSC, CSC_inv, T_DIC, optimizer, cost_function, X, images)
		return CSC, CSC_inv

		
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CSC CLASSES AND CONSISTENCY FUNCTIONS
class SL_CSC(nn.Module):
	def __init__(self, stride, dp_channels, atom_r, atom_c, numb_atom):
		super(SL_CSC, self).__init__()
		self.D_trans = nn.Conv2d(dp_channels, numb_atom, (atom_r, atom_c), stride, padding=0, dilation=1, groups=1, bias=False)
    
	def forward(self, x):
		out = self.D_trans(x)
		return out


class SL_CSC_inv(nn.Module):
	def __init__(self, stride, dp_channels, atom_r, atom_c, numb_atom):
		super(SL_CSC_inv, self).__init__()
		self.D = nn.ConvTranspose2d(numb_atom, dp_channels, (atom_c, atom_r), stride, padding=0, output_padding=0, groups=1, bias=False, dilation=1)
    
	def forward(self, x):
		out = self.D(x)
		return out

   
def make_consistent(CSC, CSC_inv):
	CSC.D_trans.weight.data = CSC_inv.D.weight.data.permute(0,1,3,2)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions for displaying images
def plot_image(img_tensor):
	print("Plotting image")
	img_numpy = img_tensor.numpy()
	plt.figure(1)
	plt.imshow(np.transpose(img_numpy, (1, 2, 0)), cmap = 'Greys')
	plt.show()

	
# MAIN LOOP
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Training Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001
T_SC = 1000
T_DIC = 5
stride = 1
learning_rate = 0.001
momentum = 0.9
num_epochs = 1

# Local dictionary dimensions
atom_r = 5
atom_c = 5
numb_atom = 100
dp_channels = 1 

# Generate synthetic training data to train model, Y is the training data or input to the classifier
# Y = Variable(torch.randn(numb_dp, dp_channels, dp_r, dp_c).type(dtype))

# Load MNIST
root = './data'
download = False  # download MNIST dataset or not

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dsets.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dsets.MNIST(root=root, train=False, transform=trans)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)


train_set_dims = list(train_set.train_data.size())
print(train_set.train_data.size())               # (60000, 28, 28)
print(train_set.train_labels.size())               # (60000)
plt.imshow(train_set.train_data[4].numpy(), cmap='gray')
plt.title('%i' % train_set.train_labels[4])
# plt.show()


# Intitilise Convolutional Sparse Coder CSC
CSC = SL_CSC(stride, dp_channels, atom_r, atom_c, numb_atom)
CSC_inv = SL_CSC_inv(stride, dp_channels, atom_r, atom_c, numb_atom)
make_consistent(CSC, CSC_inv)

# Define training settings/ options
sparse_code_method = 'FISTA'
cost_function = nn.MSELoss()
optimizer = torch.optim.SGD(CSC_inv.parameters(), lr=learning_rate, momentum=momentum)
# optimizer = torch.optim.Adam(SSC.parameters(), lr=learning_rate)

# Train Convolutional Sparse Coder
CSC, CSC_inv = train_SL_CSC(train_loader, CSC, CSC_inv, num_epochs, T_SC, T_DIC, stride, sparse_code_method, cost_function, optimizer)













