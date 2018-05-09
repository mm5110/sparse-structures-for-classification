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

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

from IHT import SL_CSC_IHT
import AuxiliaryFunctions as af
import SupportingFunctions as sf 

# Define hardware
use_cuda = True
can_use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if can_use_cuda else "cpu")
dtype = torch.float
using_azure = True
	
# MAIN LOOP
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define hardware
use_cuda = True
can_use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if can_use_cuda else "cpu")
print(device)
dtype = torch.FloatTensor

# Path to save model to
filename = "SL_CSC_IHT"

# Training hyperparameters
num_epochs = 100 #100
batch_size = 512
T_DIC = 1
stride = 1
learning_rate = 0.0007
momentum = 0.9 
weight_decay=0
k = 50
# dropout parameter
p=0.5

# Local dictionary dimensions
atom_r = 28
atom_c = 28
numb_atom = 500
dp_channels = 1 

# Load MNIST
root = './data'
download = False  # download MNIST dataset or not

# Access MNIST dataset and define processing transforms to proces
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# trans = transforms.Compose([transforms.ToTensor()])
train_set = dsets.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dsets.MNIST(root=root, train=False, transform=trans)

idx = list(range(60000))
train_sampler = SubsetRandomSampler(idx)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 sampler = train_sampler, #None
                 shuffle=False) #True


test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)


# Print out dimensions of training set
train_set_dims = list(train_set.train_data.size())
print(train_set.train_data.size())               # (60000, 28, 28)
print(train_set.train_labels.size())               # (60000)

# Intitilise Convolutional Sparse Coder CSC
CSC = SL_CSC_IHT(stride, dp_channels, atom_r, atom_c, numb_atom, k).to(device)

# Define optimisation parameters
CSC_parameters = [
{'params': CSC.D.parameters()}
]

# Define training settings/ options
cost_function = nn.MSELoss(size_average=True)

# Train Convolutional Sparse Coder
CSC = sf.train_SL_CSC(CSC, train_loader, num_epochs, T_DIC, cost_function, CSC_parameters, learning_rate, momentum, weight_decay, batch_size, p)
print("Training seqeunce finished")
filter_dims = list(np.shape(CSC.D_trans.weight.data.numpy()))

if using_azure == False:
	# Get CSC ready to process a few inputs
	CSC.batch_size = 100
	CSC.mask = torch.ones(CSC.batch_size, filter_dims[0],1,1)
	test_Y = Variable(torch.unsqueeze(test_set.test_data[:CSC.batch_size], dim=1)).type(torch.FloatTensor)/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
	#  Calculate the latent representation
	test_X, SC_error_percent, numb_SC_iterations, filters_selected = CSC.forward(test_Y)
	test_Y_recon = CSC.D(test_X)
	l2_error_percent = 100*np.sum((test_Y-test_Y_recon).data.numpy()**2)/ np.sum(test_Y.data.numpy()**2)
	id1 = 3
	id2 = 12
	id3 = 37
	# Plot original images side by side with reconstructions to get feel for how successful training was
	orig_image1 = test_Y[id1][0].data.numpy()
	orig_image2 = test_Y[id2][0].data.numpy()
	orig_image3 = test_Y[id3][0].data.numpy()
	recon_image1 = test_Y_recon[id1][0].data.numpy()
	recon_image2 = test_Y_recon[id2][0].data.numpy()
	recon_image3 = test_Y_recon[id3][0].data.numpy()
	plt.figure(5)
	plt.subplot(3,2,1)
	plt.imshow(orig_image1, cmap='gray')
	plt.title('Original Image');
	plt.subplot(3,2,2)
	plt.imshow(recon_image1, cmap='gray')
	plt.title('Reconstructed Image');
	plt.subplot(3,2,3)
	plt.imshow(orig_image2, cmap='gray')
	plt.subplot(3,2,4)
	plt.imshow(recon_image2, cmap='gray')
	plt.subplot(3,2,5)
	plt.imshow(orig_image3, cmap='gray')
	plt.subplot(3,2,6)
	plt.imshow(recon_image3, cmap='gray')
	plt.xlabel("l2 error over batch: {0:1.2f}%".format(l2_error_percent))
	plt.show(block=True)

# Save down model for future use
print("Saving model")
af.save_SL_CSC_IHT(CSC ,stride, dp_channels, atom_r, atom_c, numb_atom, filename)
print("Finished")



