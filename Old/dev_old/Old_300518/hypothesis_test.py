import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 

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
rep_batch_size = 60000
test_batch_size = 5000

# Dimension of class subspace
l = 10

# Sparsity value for pca
numb_atoms = 500
K=50

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
int_sequence = 619032
model_filename = "SL_CSC_IHT_" + str(int_sequence)
training_data_path = 'log_data/' + model_filename + '_' + 'training_log.csv'
activation_data_filename = 'log_data/' + model_filename  + '_' + 'activations.npy'

int_sequence_joint = 960271
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
filter_dims_J = list(np.shape(CSC_J.D_trans.weight.data.cpu().numpy()))

# Process one (large) batch of the training data to generate a representations
CSC_J.batch_size = rep_batch_size
CSC.batch_size = rep_batch_size
# Ensure that there is no dropout taking place
CSC.mask = (torch.ones(rep_batch_size, filter_dims[0], 1, 1))
CSC.mask = CSC.mask.to(device, dtype=dtype)
CSC_J.mask = (torch.ones(rep_batch_size, filter_dims_J[0], 1, 1))
CSC_J.mask = CSC_J.mask.to(device, dtype=dtype)

# Load data to calculate the representation for each class
train_inputs, train_labels = next(iter(train_loader))
train_inputs_mean = torch.mean(train_inputs.data, dim=0, keepdim=True)
train_inputs = train_inputs - train_inputs_mean
train_inputs = Variable(train_inputs).to(device, dtype=dtype)
train_labels = Variable(train_labels).to(device, dtype=dtype)
train_input_dims = list(train_inputs.size())

# GENERATE LOW DIM SUBSPACE FOR EACH CLASS BY EACH METHOD
# IHT
X,_,_,_ = CSC.forward(train_inputs)
# Joint IHT, normal IHT forward pass
# X_J,_,_,_ = CSC_J.forward(train_inputs)

# Joint IHT with Joint forward
_,_,_,_,_,_,X_J_cls = CSC_J.forward_training(train_inputs, train_labels, 1)

# Initialise dictionaries to store lists of tensors by class
# X_J_cls_list = {"0":[], "1":[], "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
X_cls_list = {"0":[], "1":[], "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
Y_pca_cls_list = {"0":[], "1":[], "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
# Initialise dictionaries to store tensor of data points
# X_J_cls = {}
X_cls = {}
Y_pca_cls = {}
# Initialise dictionaries to store basis vectors for each classes subspace
X_J_cls_rep = {}
X_cls_rep = {}
Y_pca_rep = {}

print("Finding representations generated by each method")
# IHT and JOINT IHT
# Sort each tensor into a dictionary of lists, one for each class
for i in range(train_input_dims[0]):
	# X_J_cls_list[str(train_labels[i].item())].append(X_J[i])
	X_cls_list[str(int(train_labels[i].item()))].append(X[i])
	Y_pca_cls_list[str(int(train_labels[i].item()))].append(train_inputs[i])

# # Take list of tensors and form stacked tensor, calculate the svd version as go along
# for key, tensor_list in X_J_cls_list.items():
# 	if len(X_J_cls_list[key]) > 0:
# 		X_J_cls[key] = torch.stack(X_J_cls_list[key], dim=0)
# 		vectorised_data_tensor = X_J_cls[key].view(X_J_cls[key].data.shape[0], X_J_cls[key].data.shape[1]*X_J_cls[key].data.shape[2]*X_J_cls[key].data.shape[3]),  
# 		vectorised_data_npy = np.asarray(vectorised_data_tensor[0])
# 		U, S, Vh = LA.svd(vectorised_data_npy.transpose(), full_matrices=True, compute_uv=True)
# 		X_J_cls_rep[key] = U[:,:l]

for key, tensor_list in X_J_cls.items():
	if len(X_J_cls[key]) > 0:
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
		X_cls_rep[key] = U[:,:l]

for key, tensor_list in Y_pca_cls_list.items():
	if len(Y_pca_cls_list[key]) > 0:
		Y_pca_cls[key] = torch.stack(Y_pca_cls_list[key], dim=0)
		vectorised_data_tensor = Y_pca_cls[key].view(Y_pca_cls[key].data.shape[0], Y_pca_cls[key].data.shape[1]*Y_pca_cls[key].data.shape[2]*Y_pca_cls[key].data.shape[3]),  
		matrix_of_inputs = np.asarray(vectorised_data_tensor[0])
		pca = PCA(n_components=l)
		pca.fit_transform(matrix_of_inputs)
		Y_pca_rep[key] = np.transpose(pca.components_)

# TEST REPRESENTATIONS OF EACH CLASS GENERATED
# Load test data
test_inputs, test_labels = next(iter(test_loader))
test_inputs_mean = torch.mean(test_inputs.data, dim=0, keepdim=True)
test_inputs = test_inputs - test_inputs_mean
test_inputs = Variable(test_inputs).to(device, dtype=dtype)
test_labels = Variable(test_labels).to(device, dtype=dtype)
test_input_dims = list(test_inputs.size())

# Process test data to find representation, we will measure how close this representation lies to the 
CSC_J.batch_size = test_batch_size
CSC.batch_size = test_batch_size
# Ensure that there is no dropout taking place
CSC.mask = (torch.ones(test_batch_size, filter_dims[0], 1, 1)).to(device, dtype=dtype)
CSC_J.mask = (torch.ones(test_batch_size, filter_dims_J[0], 1, 1)).to(device, dtype=dtype)


# ANALYSE CLASSIFICATION
# Find representation of test batch
print("Calculating representations of test set")
X_J_test,_,_,_ = CSC_J.forward(test_inputs)
test_recon_jiht = (CSC_J.D(X_J_test))
X_J_test = X_J_test.view(X_J_test.data.shape[0], X_J_test.data.shape[1]*X_J_test.data.shape[2]*X_J_test.data.shape[3]),  
X_J_test = X_J_test[0]

X_test,_,_,_ = CSC.forward(test_inputs)
test_recon_iht = (CSC.D(X_test))
X_test = X_test.view(X_test.data.shape[0], X_test.data.shape[1]*X_test.data.shape[2]*X_test.data.shape[3]),  
X_test = X_test[0]

flat_test_inputs = test_inputs.view(test_inputs.data.shape[0], test_inputs.data.shape[1]*test_inputs.data.shape[2]*test_inputs.data.shape[3])

# Initialise dictionaries that will contain the size of the projection of each data point onto each class
X_J_test_proj_mat = np.zeros((test_input_dims[0], 10))
X_test_proj_mat = np.zeros((test_input_dims[0], 10))
Y_PCA_test_proj_mat = np.zeros((test_input_dims[0], 10))

print("Calculating projections of test set onto each class vector space")
for key, tensor_list in X_J_cls_rep.items():
	if len(X_J_cls_rep[key]) > 0:
		X_J_test_proj_mat[:, int(key)] = np.sqrt(np.sum(np.matmul(X_J_test, X_J_cls_rep[key])**2, axis=1))

for key, tensor_list in X_cls_rep.items():
	if len(X_cls_rep[key]) > 0:
		X_test_proj_mat[:, int(key)] = np.sqrt(np.sum(np.matmul(X_test, X_cls_rep[key])**2, axis=1))

for key, temp in Y_pca_rep.items():
	 if len(Y_pca_rep[key]) > 0:
	 	Y_PCA_test_proj_mat[:, int(key)] = np.sqrt(np.sum(np.matmul(flat_test_inputs, Y_pca_rep[key])**2, axis=1))

# Calculate the labels by choosing the class for which data point has the largest projection
label_estimates_jiht = np.argmax(X_J_test_proj_mat, axis=1)
label_estimates_iht = np.argmax(X_test_proj_mat, axis=1)
label_estimates_pca = np.argmax(Y_PCA_test_proj_mat, axis=1)

# Calculate classification error rate in the test set
class_error_rate_jiht = 100*(1 - np.sum(np.sum(label_estimates_jiht == test_labels.data.numpy()))/test_batch_size)
class_error_rate_iht = 100*(1 - np.sum(np.sum(label_estimates_iht == test_labels.data.numpy()))/test_batch_size)
class_error_rate_pca = 100*(1 - np.sum(np.sum(label_estimates_pca  == test_labels.data.numpy()))/test_batch_size)


# ANALYSING RECONSTRUCTIONS
# Calculate representations
class_data = test_labels.data.numpy()

# Calculate reconstructions and reconstruction error of PCA approach (WIP)
pca_train_inputs = train_inputs.view(train_inputs.data.shape[0], train_inputs.data.shape[1]*train_inputs.data.shape[2]*train_inputs.data.shape[3])
pca_train_inputs = np.transpose(pca_train_inputs.data.cpu().numpy())
# print(pca_train_inputs.shape)
pca2 = PCA(n_components=numb_atoms)
pca2.fit_transform(np.transpose(pca_train_inputs))
princ_components = pca2.components_
print(princ_components.shape)

test_inner_products = np.matmul(princ_components, np.transpose(flat_test_inputs.data.cpu().numpy()))
print(test_inner_products.shape)
order = np.sort(np.abs(test_inner_products), axis=0)
Kth_largest_elements = order[K,:]
mask = test_inner_products>Kth_largest_elements
test_recon_flattened = np.matmul(np.transpose(princ_components), mask*test_inner_products)

test_recon_pca = (torch.from_numpy(np.transpose(test_recon_flattened))).view(test_batch_size, 1, 28, 28)

# Calculate the reconstruction error rate for iht methods:
test_l2 = np.sum((test_inputs**2).data.cpu().numpy())
recon_error_rate_iht = 100*np.sum(((test_inputs-test_recon_iht).data.cpu().numpy())**2)/test_l2
recon_error_rate_jiht = 100*np.sum(((test_inputs-test_recon_jiht).data.cpu().numpy())**2)/test_l2
recon_error_rate_pca = 100*np.sum(((test_inputs-test_recon_pca).data.cpu().numpy())**2)/test_l2

# Print out classification and reconstruction error rates
print("PCA - classification error rate: {0:1.2f} %".format(class_error_rate_pca) + ", reconstruction error rate: {0:1.2f} %".format(recon_error_rate_pca))
print("IHT - classification error rate: {0:1.2f} %".format(class_error_rate_iht) + ", reconstruction error rate: {0:1.2f} %".format(recon_error_rate_iht))
print("JHT - classification error rate: {0:1.2f} %".format(class_error_rate_jiht) + ", reconstruction error rate: {0:1.2f} %".format(recon_error_rate_jiht))

# OUTPUT FOR RECONSTRUCTION FOR VISUAL INSPECTION
# Convert to numpy arrays for outputting as images
test_recon_iht = test_recon_iht.data.cpu().numpy()
test_recon_jiht = test_recon_jiht.data.cpu().numpy()

# test_rep_jiht = test_rep_jiht.data.cpu().numpy()
# test_rep_iht = test_rep_iht.data.cpu().numpy()
idx = random.sample(range(0, CSC.batch_size), 3)


# WIP still need to do then PCA reconstructions in a fair manner
plt.figure(1)
plt.subplot(3,4,1)
plt.imshow(test_inputs[idx[0]][0].data.cpu().numpy(), cmap='gray')
plt.title('Original Image')
plt.xlabel("Actual class: {0:1.0f} ".format(class_data[idx[0]]))
plt.subplot(3,4,2)
plt.imshow(test_recon_pca[idx[0]][0].data.cpu().numpy(), cmap='gray')
plt.title('PCA')
plt.xlabel("Estimated class: {0:1.0f} ".format(class_data[idx[0]]))
plt.subplot(3,4,3)
plt.imshow(test_recon_iht[idx[0]][0], cmap='gray')
plt.title('IHT')
plt.xlabel("Estimated class: {0:1.0f} ".format(label_estimates_iht[idx[0]]))
plt.subplot(3,4,4)
plt.imshow(test_recon_jiht[idx[0]][0], cmap='gray')
plt.title('JIHT')
plt.xlabel("Estimated class: {0:1.0f} ".format(label_estimates_jiht[idx[0]]))
plt.subplot(3,4,5)
plt.imshow(test_inputs[idx[1]][0].data.cpu().numpy(), cmap='gray')
plt.xlabel("Actual class: {0:1.0f} ".format(class_data[idx[1]]))
plt.subplot(3,4,6)
plt.imshow(test_recon_pca[idx[1]][0].data.cpu().numpy(), cmap='gray')
plt.xlabel("Estimated class: {0:1.0f} ".format(class_data[idx[1]]))
plt.subplot(3,4,7)
plt.imshow(test_recon_iht[idx[1]][0], cmap='gray')
plt.xlabel("Estimated class: {0:1.0f} ".format(label_estimates_iht[idx[1]]))
plt.subplot(3,4,8)
plt.imshow(test_recon_jiht[idx[1]][0], cmap='gray')
plt.xlabel("Estimated class: {0:1.0f} ".format(label_estimates_jiht[idx[1]]))
plt.subplot(3,4,9)
plt.imshow(test_inputs[idx[2]][0].data.cpu().numpy(), cmap='gray')
plt.xlabel("Actual class: {0:1.0f} ".format(class_data[idx[2]]))
plt.subplot(3,4,10)
plt.imshow(test_recon_pca[idx[2]][0].data.cpu().numpy(), cmap='gray')
plt.xlabel("Estimated class: {0:1.0f} ".format(class_data[idx[2]]))
plt.subplot(3,4,11)
plt.imshow(test_recon_iht[idx[2]][0], cmap='gray')
plt.xlabel("Estimated class: {0:1.0f} ".format(label_estimates_iht[idx[2]]))
plt.subplot(3,4,12)
plt.imshow(test_recon_jiht[idx[2]][0], cmap='gray')
plt.xlabel("Estimated class: {0:1.0f} ".format(label_estimates_jiht[idx[2]]))
plt.show(block=True)

# test_input_rep,_,_,_ = CSC_J.forward(test_inputs)
# test_inputs_recon = CSC.D(test_input_rep)
# # Plot original images side by side with reconstructions to get feel for how successful training was
# orig_image1 = test_inputs[idx[0]][0].data.cpu().numpy()
# orig_image2 = test_inputs[idx[1]][0].data.cpu().numpy()
# orig_image3 = test_inputs[idx[2]][0].data.cpu().numpy()
# recon_image1 = test_inputs_recon[idx[0]][0].data.cpu().numpy()
# recon_image2 = test_inputs_recon[idx[1]][0].data.cpu().numpy()
# recon_image3 = test_inputs_recon[idx[2]][0].data.cpu().numpy()




# plt.figure(1)
# plt.subplot(3,2,1)
# plt.imshow(orig_image1, cmap='gray')
# plt.title('Original Image')
# plt.xlabel("Actual class: {0:1.0f} ".format(class_data[idx[0]]))
# plt.subplot(3,2,2)
# plt.imshow(recon_image1, cmap='gray')
# plt.title('Reconstructed Image')
# plt.xlabel("Estimated class: {0:1.0f}".format(label_estimates_jiht[idx[0]]))
# plt.subplot(3,2,3)
# plt.imshow(orig_image2, cmap='gray')
# plt.xlabel("Actual class: {0:1.0f} ".format(class_data[idx[1]]))
# plt.subplot(3,2,4)
# plt.imshow(recon_image2, cmap='gray')
# plt.xlabel("Estimated class: {0:1.0f}".format(label_estimates_jiht[idx[1]]))
# plt.subplot(3,2,5)
# plt.imshow(orig_image3, cmap='gray')
# plt.xlabel("Actual class: {0:1.0f} ".format(class_data[idx[2]]))
# plt.subplot(3,2,6)
# plt.imshow(recon_image3, cmap='gray')
# plt.xlabel("Estimated class: {0:1.0f}".format(label_estimates_jiht[idx[2]]))
# plt.show(block=True)




