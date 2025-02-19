import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import time
import pdb
import models as mds

import random
import os
import yaml
import datetime
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


####################################
    ## Save and load functions ##
####################################

def save_model(model, filename):
	# Define paths
	torch_save_path = os.getcwd() + "/trained_models/" + filename + ".pt"
	yml_save_path = os.getcwd() + "/trained_models/" + filename + ".yml"
	# Save down weight matrix
	torch.save(model.W, torch_save_path)
	# Define dictionary with model's other vairbales
	other_CSC_variable_data = {}
	other_CSC_variable_data["K"] = model.K
	other_CSC_variable_data["forward_type"] = model.forward_type
	other_CSC_variable_data["m"] = model.m
	# Save down dictionary in a yaml file
	with open(yml_save_path, 'w') as yaml_file:
	    yaml.dump(other_CSC_variable_data, stream=yaml_file, default_flow_style=False)


def load_model(filename):
	torch_load_path = os.getcwd() + "/trained_models/" + filename + ".pt"
	yml_load_path = os.getcwd() + "/trained_models/" + filename + ".yml"
	# Load in model
	with open(yml_load_path, 'r') as yaml_file:
	    loaded_model_vars = yaml.load(yaml_file)
	# Initialise and return CSC
	if loaded_model_vars["forward_type"] == "IHT":
		model = mds.DictLearnt_IHT(int(loaded_model_vars["m"]), int(loaded_model_vars["K"]))
	elif loaded_model_vars["forward_type"] == "JIHT":
		model = mds.DictLearnt_JIHT(int(loaded_model_vars["m"]), int(loaded_model_vars["K"]))
	else:
		print("Error, forward type not recognised: " + loaded_model_vars["forward_type"])	
	# Load in model parameters
	temp = torch.load(torch_load_path)
	model.W.data = torch.load(torch_load_path)
	# Return model 
	return model