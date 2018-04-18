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

# Provide data filename (both yml and pt file) in which the target model data is stored
filename = "SL_CSC_IHT"

# Testing parameters
batch_size=10

# Load MNIST
root = './data'
download = False  # download MNIST dataset or not

# Access MNIST dataset and define processing transforms to proces
# trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
trans = transforms.Compose([transforms.ToTensor()])
test_set = dsets.MNIST(root=root, train=False, transform=trans)

test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

# # Load in model
CSC = scc.load_SL_CSC_IHT(filename)

# Test reconstruction capabilities of trained CSC, first extract some test examples
test_Y = Variable(torch.unsqueeze(test_set.test_data, dim=1), volatile=True).type(torch.FloatTensor)/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_Y =Variable(test_Y.data[:10])
#  Calculate the latent representation
test_X = CSC.forward(test_Y)
test_Y_recon = CSC.reverse(test_X)

id1 = 2
id2 = 7
id3 = 9

# Plot original images side by side with reconstructions to get feel for how successful training was
orig_image1 = test_Y[id1][0].data.numpy()
orig_image2 = test_Y[id2][0].data.numpy()
orig_image3 = test_Y[id3][0].data.numpy()
recon_image1 = test_Y_recon[id1][0].data.numpy()
recon_image2 = test_Y_recon[id2][0].data.numpy()
recon_image3 = test_Y_recon[id3][0].data.numpy()

plt.figure(1)
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
plt.show()




