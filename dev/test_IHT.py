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

import AuxiliaryFunctions as af
import SupportingFunctions as sf 


# Provide data filename (both yml and pt file) in which the target model data is stored
int_sequence = 385366
model_filename = "SL_CSC_IHT_" + str(int_sequence)
training_data_path = 'log_data/' + model_filename + '_' + 'training_log.csv'
activation_data_filename = 'log_data/' + model_filename  + '_' + 'activations.npy'

# Testing parameters
batch_size=100

# Load MNIST
root = './data'
download = False  # download MNIST dataset or not

# Access MNIST dataset and define processing transforms to proces
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_set = dsets.MNIST(root=root, train=False, transform=trans)

test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

# Load in model
CSC = af.load_SL_CSC_IHT(model_filename)
CSC.batch_size = batch_size
filter_dims = list(np.shape(CSC.D_trans.weight.data.cpu().numpy()))
CSC.mask = torch.ones(batch_size, filter_dims[0], 1,1)

# View training training data
training_data = np.loadtxt(training_data_path, delimiter=',',skiprows=1)
plt.figure(1)
plt.plot(training_data[:,2], training_data[:,5])
plt.title("Training error over time")
plt.ylabel("Reconstruction percentage error")
plt.xlabel("Number of batches")
plt.show()

plt.figure(2)
plt.plot(training_data[:,2], training_data[:,4])
plt.title("Number of IHT iterations")
plt.ylabel("Number of sparse coding iterations")
plt.xlabel("Number of batches")
plt.show()

# Plot all filters at the end of the training sequence
D = CSC.D.weight.data.cpu().numpy()
M = af.showFilters(D,20,20)
plt.figure(3, figsize=(10,10))
plt.imshow(rescale(M, scale=4, mode='constant'),cmap='gray')
plt.axis('off')
plt.show()

# Load in histogram data
filter_activations = np.load(activation_data_filename)
plt.figure(4)
plt.hist(filter_activations, bins=50)  # arguments are passed to np.histogram
plt.title("Histogram of filter activations")
plt.ylabel("Number of filters")
plt.xlabel("Activation frequency bin")
plt.show()

# Test reconstruction capabilities of trained CSC, first extract some test examples
test_Y = Variable(torch.unsqueeze(test_set.test_data[:batch_size], dim=1)).type(torch.FloatTensor)/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
#  Calculate the latent representation
test_X, SC_error_percent, numb_SC_iterations, filters_selected = CSC.forward(test_Y)
test_Y_recon = CSC.D(test_X)
l2_error_percent = 100*np.sum((test_Y-test_Y_recon).data.cpu().numpy()**2)/ np.sum(test_Y.data.cpu().numpy()**2)

idx = random.sample(range(0, batch_size), 3)
# Plot original images side by side with reconstructions to get feel for how successful training was
orig_image1 = test_Y[idx[0]][0].data.cpu().numpy()
orig_image2 = test_Y[idx[1]][0].data.cpu().numpy()
orig_image3 = test_Y[idx[2]][0].data.cpu().numpy()
recon_image1 = test_Y_recon[idx[0]][0].data.cpu().numpy()
recon_image2 = test_Y_recon[idx[1]][0].data.cpu().numpy()
recon_image3 = test_Y_recon[idx[2]][0].data.cpu().numpy()

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
plt.show()

