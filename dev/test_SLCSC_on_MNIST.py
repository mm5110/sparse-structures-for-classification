import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

import sparse_coding_classifier_functions as scc

load_path = os.getcwd() + "/trained_models/SL_CSC_FISTA.pt"

# Training hyperparameters
num_epochs = 1
batch_size = 50
T_SC = 50
T_DIC = 10
T_PM = 8
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

# Define cost function on which to evaluate performance
cost_function = nn.MSELoss()

# Load MNIST
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

# Load in model
CSC = scc.SL_CSC_FISTA(stride, dp_channels, atom_r, atom_c, numb_atom, 1, T_SC, T_PM)
print(CSC.tau)
CSC.load_state_dict(torch.load(load_path))
print(CSC.tau)

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