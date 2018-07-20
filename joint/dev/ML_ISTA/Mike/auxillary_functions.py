import numpy as np
import torch

def compute_histogram(gamma, m3, b_y):
    activations = np.zeros((10, m3))
    for i in range(10):
        n_ci = float(torch.sum(b_y==i))
        if n_ci>0:
            gamma_i = gamma[b_y[b_y==i],:,:,:]
            activations[i,:] = count_activations(gamma_i, 1e-4)
    return activations


def count_activations(gamma_i, tol=1e-4):
    gamma_dims = list(gamma_i.shape)
    gamma_i[gamma_i<tol]=0
    gamma_i[gamma_i>=tol]=1
    if len(gamma_dims)>3:
        activations = np.sum(gamma_i, axis=3)
        activations = np.sum(activations, axis=2)
        activations = np.sum(activations, axis=0)
    else:
        activations = np.sum(gamma_i, axis=2)
        activations = np.sum(activations, axis=1)
    return activations