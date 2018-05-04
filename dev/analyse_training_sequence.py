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

# Define training sequence wish to analyse
numpy_data_file = "/Users/mmurray/Documents/github/my-projects/sparse-structures-for-classification/dev/log_data/2018-05-04 15/46/33.925616_IHT_activations.npy"
csv_training_data_file = "/Users/mmurray/Documents/github/my-projects/sparse-structures-for-classification/dev/log_data/2018-05-04 15/46/33.925616_IHT_training_log.csv"


# Load in activation data at the end of training
filter_activations = np.load(numpy_data_file)

# Plot histogram of activations
plt.figure(1)
plt.hist(filter_activations, bins=50)  # arguments are passed to np.histogram
plt.title("Histogram of filter activations, epoch Number: {0:1.0f}".format(epoch) + ", batch number: {0:1.0f}".format(i+1))
plt.ylabel("Number of filters")
plt.xlabel("Activation frequency bin")
plt.draw()
plt.pause(0.001)



