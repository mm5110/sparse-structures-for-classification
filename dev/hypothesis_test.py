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

# Define hardware
use_cuda = True
can_use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if can_use_cuda else "cpu")
print(device)
dtype = torch.float
using_azure = False

# Parameters
rep_batch_size = 1000
test_batch_size = 10
l = 50

# Load MNIST
root = './data'
download = True  # download MNIST dataset or not

# Access MNIST dataset and define processing transforms to proces
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# trans = transforms.Compose([transforms.ToTensor()])
train_set = dsets.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dsets.MNIST(root=root, train=False, transform=trans)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=rep_batch_size,
                 sampler = None,
                 shuffle=True)


test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=test_batch_size,
                shuffle=True)

# Provide data filename (both yml and pt file) in which the target model data is stored
int_sequence = 647744
model_filename = "SL_CSC_IHT_" + str(int_sequence)
training_data_path = 'log_data/' + model_filename + '_' + 'training_log.csv'
activation_data_filename = 'log_data/' + model_filename  + '_' + 'activations.npy'

int_sequence_joint = 202083
model_filename_joint = "SL_CSC_IHT_joint_" + str(int_sequence_joint)
training_data_path_joint = 'log_data/' + model_filename_joint + '_' + 'training_log.csv'
activation_data_filename_joint = 'log_data/' + model_filename_joint  + '_' + 'activations.npy'

# Load in model IHT model
print("Loading CSC IHT model")
CSC = af.load_SL_CSC_IHT(model_filename)
filter_dims = list(np.shape(CSC.D_trans.weight.data.cpu().numpy()))

# Load in model Joint IHT model
print("Loading CSC Joint IHT model")
CSC_J = af.load_SL_CSC_IHT(model_filename_joint)
filter_dims_J = list(np.shape(CSC.D_trans.weight.data.cpu().numpy()))

# Process one (large) batch of the training data to generate a representations
CSC_J.batch_size = rep_batch_size
CSC.batch_size = rep_batch_size
# Ensure that there is no dropout taking place
CSC.mask = torch.ones(rep_batch_size, filter_dims[0], 1, 1)
CSC_J.mask = torch.ones(rep_batch_size, filter_dims_J[0], 1, 1)

# Load data to calculate the representation for each class
train_inputs, train_labels = next(iter(train_loader))
train_inputs = Variable(train_inputs).to(device)
train_labels = Variable(train_labels).to(device)
train_input_dims = list(train_inputs.size())

# Generate Representations
X_J,_,_,_ = CSC_J.forward(train_inputs)
X,_,_,_ = CSC.forward(train_inputs)

# Initialise dictionaries
X_J_cls_list = {"0":[], "1":[], "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
X_cls_list = {"0":[], "1":[], "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
X_J_cls = {}
X_cls = {}
X_J_cls_rep = {}
X_cls_rep = {}

print("Finding joint representation of each class using svd")
# Sort each tensor into a dicitonary of lists, one for each class
for i in range(train_input_dims[0]):
	X_J_cls_list[str(train_labels[i].item())].append(X_J[i])
	X_cls_list[str(train_labels[i].item())].append(X[i])

# Take list of tensors and form stacked tensor, calculate the svd version as go along
for key, tensor_list in X_J_cls_list.items():
	if len(X_J_cls_list[key]) > 0:
		X_J_cls[key] = torch.stack(X_J_cls_list[key], dim=0)
		vectorised_data_tensor = X_J_cls[key].view(X_J_cls[key].data.shape[0], X_J_cls[key].data.shape[1]*X_J_cls[key].data.shape[2]*X_J_cls[key].data.shape[3]),  
		vectorised_data_npy = np.asarray(vectorised_data_tensor[0])
		U, S, Vh = LA.svd(vectorised_data_npy.transpose(), full_matrices=True, compute_uv=True)
		X_J_cls_rep[key] = U[:,:l] 

for key, tensor_list in X_cls_list.items():
	if len(X_cls_list[key]) > 0:
		X_cls[key] = torch.stack(X_cls_list[key], dim=0)
		vectorised_data_tensor = X_cls[key].view(X_cls[key].data.shape[0], X_cls[key].data.shape[1]*X_cls[key].data.shape[2]*X_cls[key].data.shape[3]),  
		vectorised_data_npy = np.asarray(vectorised_data_tensor[0])
		U, S, Vh = LA.svd(vectorised_data_npy.transpose(), full_matrices=True, compute_uv=True)
		X_J_cls_rep[key] = U[:,:l]

# Load test data
test_inputs, test_labels = next(iter(test_loader))
test_inputs = Variable(test_inputs).to(device)
test_labels = Variable(test_labels).to(device)
test_input_dims = list(test_inputs.size())

# Process one (large) batch of the training data to generate a representations
CSC_J.batch_size = test_batch_size
CSC.batch_size = test_batch_size
# Ensure that there is no dropout taking place
CSC.mask = torch.ones(test_batch_size, filter_dims[0], 1, 1)
CSC_J.mask = torch.ones(test_batch_size, filter_dims_J[0], 1, 1)

# Find representation of test batch
print("Calculating representations of test set")
X_J_test,_,_,_ = CSC_J.forward(test_inputs)
X_J_test = X_J_test.view(X_J_test.data.shape[0], X_J_test.data.shape[1]*X_J_test.data.shape[2]*X_J_test.data.shape[3]),  
X_J_test = X_J_test[0]
X_test,_,_,_ = CSC.forward(test_inputs)
X_test = X_test.view(X_test.data.shape[0], X_test.data.shape[1]*X_test.data.shape[2]*X_test.data.shape[3]),  
X_test = X_test[0]

# Initialise dictionaries that will contain the size of the projection of each data point onto each class
X_J_test_proj_mat = np.zeros((test_input_dims[0], 10))
X_test_proj_mat = np.zeros((test_input_dims[0], 10))

print("Calculating projections of test set onto each class vector space")
for key, tensor_list in X_J_cls_rep.items():
	if len(X_J_cls_rep[key]) > 0:
		print(np.shape(X_J_test))
		print(np.shape(X_J_cls_rep[key]))
		X_J_test_proj_mat[:, int(key)] = np.sqrt(np.sum(np.matmul(X_J_test, X_J_cls_rep[key])**2, axis=1))

for key, tensor_list in X_cls_rep.items():
	if len(X_cls_rep[key]) > 0:
		X_test_proj_mat[:, int(key)] = np.sqrt(np.sum(np.matmul(X_test, X_cls_rep[key])**2, axis=1))

# Calculate the labels by choosing the class for which data point has the largest projection
label_estimates_joint = np.argmax(X_J_test_proj_mat, axis=1)
label_estimates = np.argmax(X_test_proj_mat, axis=1)

# Calculate classification error rate in the test set
joint_error_rate = 100*(1 - np.sum(np.sum(label_estimates_joint == test_labels.data.numpy()))/test_batch_size)
error_rate = 100*(1 - np.sum(np.sum(label_estimates == test_labels.data.numpy()))/test_batch_size)

print("Joint IHT error rate: {0:1.2f} %".format(joint_error_rate))
print("IHT error rate: {0:1.2f} %".format(error_rate))

print(X_J_test_proj_mat)
print(label_estimates_joint)
print(test_labels.data.numpy())


test_input_rep,_,_,_ = CSC_J.forward(test_inputs)
test_inputs_recon = CSC_J.D(test_input_rep)
idx = random.sample(range(0, CSC.batch_size), 3)
# Plot original images side by side with reconstructions to get feel for how successful training was
orig_image1 = test_inputs[idx[0]][0].data.cpu().numpy()
orig_image2 = test_inputs[idx[1]][0].data.cpu().numpy()
orig_image3 = test_inputs[idx[2]][0].data.cpu().numpy()
recon_image1 = test_inputs_recon[idx[0]][0].data.cpu().numpy()
recon_image2 = test_inputs_recon[idx[1]][0].data.cpu().numpy()
recon_image3 = test_inputs_recon[idx[2]][0].data.cpu().numpy()


class_data = test_labels.data.numpy()

plt.figure(1)
plt.subplot(3,2,1)
plt.imshow(orig_image1, cmap='gray')
plt.title('Original Image')
plt.xlabel("Actual class: {0:1.0f} ".format(class_data[idx[0]]))
plt.subplot(3,2,2)
plt.imshow(recon_image1, cmap='gray')
plt.title('Reconstructed Image')
plt.xlabel("Estimated class: {0:1.0f}".format(label_estimates_joint[idx[0]]))
plt.subplot(3,2,3)
plt.imshow(orig_image2, cmap='gray')
plt.xlabel("Actual class: {0:1.0f} ".format(class_data[idx[1]]))
plt.subplot(3,2,4)
plt.imshow(recon_image2, cmap='gray')
plt.xlabel("Estimated class: {0:1.0f}".format(label_estimates_joint[idx[1]]))
plt.subplot(3,2,5)
plt.imshow(orig_image3, cmap='gray')
plt.xlabel("Actual class: {0:1.0f} ".format(class_data[idx[2]]))
plt.subplot(3,2,6)
plt.imshow(recon_image3, cmap='gray')
plt.xlabel("Estimated class: {0:1.0f}".format(label_estimates_joint[idx[2]]))
plt.show(block=True)





