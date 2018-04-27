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

import sparse_coding_classifier_functions as scc

from AuxiliaryFunctions import showFilters
from skimage.transform import rescale, resize, downscale_local_mean

	
# MAIN LOOP
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Path to save model to
filename = "SL_CSC_OMP"

# Training hyperparameters
num_epochs = 1 #100
batch_size = 1
T_SC = 40
T_DIC = 100
stride = 1
learning_rate = 1
momentum = 0.9
weight_decay=0
k = 5

# Local dictionary dimensions
atom_r = 28
atom_c = 28
numb_atom = 15
dp_channels = 1 

# Load MNIST
root = './data'
download = False  # download MNIST dataset or not

# Access MNIST dataset and define processing transforms to proces
# trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
trans = transforms.Compose([transforms.ToTensor()])
train_set = dsets.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dsets.MNIST(root=root, train=False, transform=trans)

idx = list(range(10000))
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
CSC = scc.SL_CSC_OMP(stride, dp_channels, atom_r, atom_c, numb_atom, T_SC, k)

# Define optimisation parameters
CSC_parameters = [
{'params': CSC.D.parameters()}
]

# Define training settings/ options
cost_function = nn.MSELoss(size_average=True)
optimizer = torch.optim.SGD(CSC_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)
# optimizer = torch.optim.Adam(SSC.parameters(), lr=learning_rate)

# Train Convolutional Sparse Coder
CSC = scc.train_SL_CSC(CSC, train_loader, num_epochs, T_DIC, cost_function, optimizer, batch_size)
print("Training seqeunce finished")


print("Plotting learned filters after training")
# Plot all filters at the end of the training sequence
D = CSC.D.weight.data.numpy()
M = showFilters(D,10,10)
plt.figure(5, figsize=(30,30))
plt.imshow(rescale(M, scale =4, mode='constant'),cmap='gray')
plt.axis('off')
plt.show(block = True)

print("Testing model on a few images from the training set")
# Test reconstruction capabilities of trained CSC, first extract some test examples
test_Y = Variable(torch.unsqueeze(test_set.test_data, dim=1), volatile=True).type(torch.FloatTensor)/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_Y =Variable(test_Y.data[:3])
#  Calculate the latent representation
test_X = CSC.forward(test_Y)
test_Y_recon = CSC.reverse(test_X)
# Plot original images side by side with reconstructions to get feel for how successful training was
orig_image1 = test_Y[0][0].data.numpy()
orig_image2 = test_Y[1][0].data.numpy()
orig_image3 = test_Y[2][0].data.numpy()
recon_image1 = test_Y_recon[0][0].data.numpy()
recon_image2 = test_Y_recon[1][0].data.numpy()
recon_image3 = test_Y_recon[2][0].data.numpy()
plt.figure(6)
plt.subplot(3,2,1)
plt.imshow(orig_image1, cmap='gray')
plt.title('Original Image')
plt.subplot(3,2,2)
plt.imshow(recon_image1, cmap='gray')
plt.title('Reconstructed Image')
plt.subplot(3,2,3)
plt.imshow(orig_image2, cmap='gray')
plt.subplot(3,2,4)
plt.imshow(recon_image2, cmap='gray')
plt.subplot(3,2,5)
plt.imshow(orig_image3, cmap='gray')
plt.subplot(3,2,6)
plt.imshow(recon_image3, cmap='gray')
plt.show(block = True)

# Save down model for future use
print("Saving model")
scc.save_SL_CSC_IHT(CSC ,stride, dp_channels, atom_r, atom_c, numb_atom, filename)
print("Finished")



