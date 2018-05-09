import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

import random
import os
import yaml
import datetime
import csv

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

import SupportingFunctions as sf
import IHT as IHT

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DISPLAY FUNCTIONS
def showFilters(W,ncol,nrows):
    # Display filters
    p = int(W.shape[3])+2
    Nimages = W.shape[0]
    Mosaic = np.zeros((p*ncol,p*nrows))
    indx = 0
    for i in range(ncol):
        for j in range(nrows):
            im = W[indx,0,:,:]
            im = (im-np.min(im))
            im = im/np.max(im)
            Mosaic[ i*p : (i+1)*p , j*p : (j+1)*p ] = np.pad(im.reshape(W.shape[3],W.shape[3]),(1,1),mode='constant').reshape(p,p)
            indx += 1
            
    return Mosaic

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SAVE AND LOAD FUNCTIONS
def save_SL_CSC_IHT(CSC, stride, dp_channels, atom_r, atom_c, numb_atom, filename):
    torch_save_path = os.getcwd() + "/trained_models/" + filename + ".pt"
    yml_save_path = os.getcwd() + "/trained_models/" + filename + ".yml"
    # Save model parameters
    torch.save(CSC.state_dict(), torch_save_path)
    # Define dictionary of other variables to store
    other_CSC_variable_data = {}
    other_CSC_variable_data["stride"] = stride
    other_CSC_variable_data["dp_channels"] = dp_channels
    other_CSC_variable_data["atom_r"] = atom_r
    other_CSC_variable_data["atom_c"] = atom_c
    other_CSC_variable_data["numb_atom"] = numb_atom
    other_CSC_variable_data["k"] = CSC.k
    # Save down dictionary in a yaml file
    with open(yml_save_path, 'w') as yaml_file:
        yaml.dump(other_CSC_variable_data, stream=yaml_file, default_flow_style=False)

def load_SL_CSC_IHT(filename):
    torch_load_path = os.getcwd() + "/trained_models/" + filename + ".pt"
    yml_load_path = os.getcwd() + "/trained_models/" + filename + ".yml"
    # Load in model
    with open(yml_load_path, 'r') as yaml_file:
        loaded_CSC_vars = yaml.load(yaml_file)
    # Initialise and return CSC
    CSC = IHT.SL_CSC_IHT(loaded_CSC_vars["stride"], loaded_CSC_vars["dp_channels"], loaded_CSC_vars["atom_r"], loaded_CSC_vars["atom_c"], loaded_CSC_vars["numb_atom"], loaded_CSC_vars["k"])
    # Load in network parameters
    CSC.load_state_dict(torch.load(torch_load_path))
    # Return model 
    return CSC

def save_SL_CSC_FISTA(CSC, stride, dp_channels, atom_r, atom_c, numb_atom, filename):
    torch_save_path = os.getcwd() + "/trained_models/" + filename + ".pt"
    yml_save_path = os.getcwd() + "/trained_models/" + filename + ".yml"
    # Save model parameters
    torch.save(CSC.state_dict(), torch_save_path)
    # Define dictionary of other variables to store
    other_CSC_variable_data = {}
    other_CSC_variable_data["stride"] = stride
    other_CSC_variable_data["dp_channels"] = dp_channels
    other_CSC_variable_data["atom_r"] = atom_r
    other_CSC_variable_data["atom_c"] = atom_c
    other_CSC_variable_data["numb_atom"] = numb_atom
    other_CSC_variable_data["tau"] = CSC.tau
    other_CSC_variable_data["T_SC"] = CSC.T_SC
    other_CSC_variable_data["T_PM"] = CSC.T_PM
    # Save down dictionary in a yaml file
    with open(yml_save_path, 'w') as yaml_file:
        yaml.dump(other_CSC_variable_data, stream=yaml_file, default_flow_style=False)

def load_SL_CSC_FISTA(filename):
    torch_load_path = os.getcwd() + "/trained_models/" + filename + ".pt"
    yml_load_path = os.getcwd() + "/trained_models/" + filename + ".yml"

    # Load in model
    with open(yml_load_path, 'r') as yaml_file:
        loaded_CSC_vars = yaml.load(yaml_file)

    # Initialise and return CSC
    CSC = SL_CSC_FISTA_backtracking(loaded_CSC_vars["stride"], loaded_CSC_vars["dp_channels"], loaded_CSC_vars["atom_r"], loaded_CSC_vars["atom_c"], loaded_CSC_vars["numb_atom"], loaded_CSC_vars["tau"], loaded_CSC_vars["T_SC"], loaded_CSC_vars["T_PM"])
    # Load in network parameters
    CSC.load_state_dict(torch.load(torch_load_path))
    # Return model 
    return CSC

def log_training_data(log_file, initialise, log_data, fieldnames):
    if initialise == True:
        with open(log_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({fieldnames[0]: log_data[0], fieldnames[1]: log_data[1], fieldnames[2]: log_data[2], fieldnames[3]: log_data[3], fieldnames[4]: log_data[4], fieldnames[5]: log_data[5], fieldnames[6]: log_data[6], fieldnames[7]: log_data[7]})
    else:
        with open(log_file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({fieldnames[0]: log_data[0], fieldnames[1]: log_data[1], fieldnames[2]: log_data[2], fieldnames[3]: log_data[3], fieldnames[4]: log_data[4], fieldnames[5]: log_data[5], fieldnames[6]: log_data[6], fieldnames[7]: log_data[7]})