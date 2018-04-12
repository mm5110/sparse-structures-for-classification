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
from torch.utils.data.sampler import SubsetRandomSampler

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
	X = Variable(torch.randn(1, w_dims[0], (y_dims[2]-w_dims[2]+1),(y_dims[3]-w_dims[3]+1)))
	# Normalise initialised vector in the l2 norm
	X = X/(np.asscalar(np.sum((X**2).data.numpy()))**0.5)
	# Perform T_PM iterations applying A^TA to x then normalising
	for i in range(T_PM):
		X = CSC.D_trans(CSC_inv.D(X))
		X = X/(np.asscalar(np.sum((X**2).data.numpy()))**0.5)
	# Compute dominant singular value
	L = (np.asscalar(np.sum((CSC.D_trans(CSC_inv.D(X))**2).data.numpy())))**0.5
	print("L value calculated: " + repr(L))
	return L

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PRIMARY ALGORITHM FUNCTIONS
def update_dictionary(CSC, CSC_inv, T_DIC, optimizer, cost_function, X, Y):
	print("Running dictionary update")
	# Update weight matrix
	for i in range(T_DIC):
		# Zero the gradient
		optimizer.zero_grad()
		# Calculate estimate of reconstructed Y
		Y_recon = CSC_inv.reverse(X)
		# Calculate loss according to the defined cost function between the true Y and reconstructed Y
		loss = cost_function(Y_recon, Y)
		if (i+1)%20 == 0:
			print("Average loss per data point at iteration " +repr(i+1) + " :" + repr(np.asscalar(loss.data.numpy())))
		# Calculate the gradient of the cost function wrt to each parameters
		loss.backward()
		# Update each parameter according to the optimizer update rule (single step)
		optimizer.step()
	# Ensure that weights for the CL-CSC and inverse CL-CSC are consistent	
	CSC.D_trans.weight.data = CSC_inv.D.weight.data.permute(0,1,3,2)


def train_SL_CSC(train_loader, CSC, CSC_inv, num_epochs, T_SC, tau, T_DIC, stride, sparse_code_method, cost_function, optimizer):
	print("Training SL-CSC. Batch size: ")
	for epoch in range(num_epochs):
		print("Training Epoch: " + repr(epoch+1))
		for i, (images, labels) in enumerate(train_loader):
			print("Batch number " + repr(i+1))
			images = Variable(images)
			labels = Variable(labels)
			# Calculate step size for sparse coding step
			step_size = tau/power_method(CSC, CSC_inv, 10, images)
			# Fix dictionary and calculate sparse code
			X = CSC.forward(images, CSC_inv, T_SC, step_size, sparse_code_method)
			# Fix sparse code and update dictionary
			update_dictionary(CSC, CSC_inv, T_DIC, optimizer, cost_function, X, images)
		
	return CSC, CSC_inv

		
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CSC CLASSES AND CONSISTENCY FUNCTIONS
class SL_CSC(nn.Module):
	def __init__(self, stride, dp_channels, atom_r, atom_c, numb_atom):
		super(SL_CSC, self).__init__()
		self.D_trans = nn.Conv2d(dp_channels, numb_atom, (atom_r, atom_c), stride, padding=0, dilation=1, groups=1, bias=False)

	def forward(self, Y, CSC_inv, T, step_size, sparse_code_method):
		# Generate latent representation
		# Process according to sparse recovery method
		if sparse_code_method == 'FISTA':
			print("Running FISTA to recover/ estimate sparse code")
			# Initialise t variables needed for FISTA
			t2 = 1
			# Initialise X1 Variable - note we need X1 and X2 as we need to use the prior two prior estimates for each update
			y_dims = list(Y.data.size())
			w_dims = list(self.D_trans.weight.data.size())
			X1 = Variable(torch.randn(y_dims[0], w_dims[0], (y_dims[2]-w_dims[2]+1),(y_dims[3]-w_dims[3]+1)))
			# Minimizer argument
			ST_arg = X1 - step_size*self.D_trans(CSC_inv.D(X1)-Y)
			for i in range(0,T):
				# Calculate latest sparse code estimate
				X2 = soft_thresh(ST_arg, step_size)
				if (i+1)%10 == 0:
					print("Iteration: "+repr(i+1)+ ", Sparsity level: " +repr(X2.data.nonzero().numpy().shape[0]))
				# If this was not the last iteration then update variables needed for the next iteration
				if i <T:
					# Update t variables
					t1 = t2
					t2 = (1 + np.sqrt(1+4*(t1**2)))/2 
					# Update Z
					Z = X2 + (X2-X1)*(t1 - 1)/t2
					# Construct minimizer argument to feed into the soft thresholding function
					ST_arg = Z - step_size*self.D_trans((CSC_inv.D(Z)-Y))
					# Update X1 to calculate next Z value
					X1 = X2

		# Return sparse representation
		return X2

class SL_CSC_inv(nn.Module):
	def __init__(self, stride, dp_channels, atom_r, atom_c, numb_atom):
		super(SL_CSC_inv, self).__init__()
		self.D = nn.ConvTranspose2d(numb_atom, dp_channels, (atom_c, atom_r), stride, padding=0, output_padding=0, groups=1, bias=False, dilation=1)
    
	def reverse(self, x):
		out = self.D(x)
		return out


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions for displaying images

	
# MAIN LOOP
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Training Parameters
num_epochs = 3
batch_size = 50
T_SC = 50
T_DIC = 100
stride = 1
learning_rate = 0.001
momentum = 0.9
num_epochs = 1

# Set regulatrisation parameter for FISTA sparse coding step, set between 0 and 1 
tau = 0.001

# Local dictionary dimensions
atom_r = 7
atom_c = 7
numb_atom = 25
dp_channels = 1 

# Generate synthetic training data to train model, Y is the training data or input to the classifier
# Y = Variable(torch.randn(numb_dp, dp_channels, dp_r, dp_c).type(dtype))

# Load MNIST
root = './data'
download = False  # download MNIST dataset or not

# trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
trans = transforms.Compose([transforms.ToTensor()])
train_set = dsets.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dsets.MNIST(root=root, train=False, transform=trans)

idx = list(range(1000))
train_sampler = SubsetRandomSampler(idx)
print(train_sampler)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 sampler = train_sampler,
                 shuffle=False)


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
# Ensure that forward and reverse nets share the same weights
CSC.D_trans.weight.data = CSC_inv.D.weight.data.permute(0,1,3,2)

# Define training settings/ options
sparse_code_method = 'FISTA'
cost_function = nn.MSELoss()
optimizer = torch.optim.SGD(CSC_inv.parameters(), lr=learning_rate, momentum=momentum)
# optimizer = torch.optim.Adam(SSC.parameters(), lr=learning_rate)

# Train Convolutional Sparse Coder
CSC, CSC_inv = train_SL_CSC(train_loader, CSC, CSC_inv, num_epochs, T_SC, tau, T_DIC, stride, sparse_code_method, cost_function, optimizer)



# Test CSC
test_Y = Variable(torch.unsqueeze(test_set.test_data, dim=1), volatile=True).type(torch.FloatTensor)/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_Y =Variable(test_Y.data[:3])
print(list(test_Y.size()))
step_size = tau/power_method(CSC, CSC_inv, 10, test_Y)
test_X = CSC.forward(test_Y, CSC_inv, T_SC, step_size, sparse_code_method)
test_Y_recon = CSC_inv.reverse(test_X)

plot_index = 2
print(test_Y[plot_index][0])
orig_image = test_Y[plot_index][0].data.numpy()
recon_image = test_Y_recon[plot_index][0].data.numpy()



plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(orig_image, cmap='gray')
plt.title('Original Image');
plt.subplot(1,2,2)
plt.imshow(recon_image, cmap='gray')
plt.title('Reconstructed Image');
plt.show()






