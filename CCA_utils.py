# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as linalg

def mapVisualFeatures(V) :
    return V

def mapTagFeatures(T) :
    return T
    
def mapClassFeatures(C) :
    return C

def computeCovMatrix(mapped_features) :
    # Takes the list of mapped_features (phi(X_i)) as input (two or
    # three elements respectively for the two and three-view CCA) and
    # returns :
    # The covariance matrix S composed of all pairs of covariance
    # matrices between the different views, and S_D, the bloc
    # diagonal matrix composed of the self-covariance matrices
    # for each view
    
    n_views = len(mapped_features)
    dimensions = np.zeros((1, n_views), dtype=np.int)
    
    for i in range(n_views) :
        dimensions[0, i] = (mapped_features[i]).shape[1]

    dim = np.sum(dimensions)
    S = np.zeros((dim, dim))
    S_D = np.zeros((dim, dim))
    
    indices = np.append(0, dimensions)
    indices = np.cumsum(indices)
    
    for i in range(n_views) :
        for j in range(i) :
            S_ij = np.dot(mapped_features[i].T, mapped_features[j])
            S[indices[i]:indices[i+1], indices[j]:indices[j+1]] = S_ij
        
        S_ii = np.dot(mapped_features[i].T, mapped_features[i])
        S_D[indices[i]:indices[i+1], indices[i]:indices[i+1]] = S_ii

    S = S + S.T + S_D
    return S, S_D
        

def solveCCA(S, S_D, d, regularization, p) :
    ## inputs :
    # S : the global covariance matrix between all pairs of "views"
    # S_D : the bloc diagonal matrix composed of the self-covariance
    # matrices for each view
    # d : dimension of the common CCA space (number of kept
    # eigenvectors and eigenvalues)
    # regularization : added to the diagonal of the covariance matrix
    # in order to regularize the problem
    # p : the power to each the eigenvalues are elevated in the
    # output D
    
    ## outputs :
    # W : the matrix composed of the d eigenvectors as columns
    # D : diagonal matix given by the p-th power of the d
    # corresponding eigenvalues
    
    I_g = regularization * np.eye(len(S))
    S = S + I_g
    S_D = S_D + I_g
    
    eigenValues, eigenVectors = linalg.eig(S, S_D)
    
    # Get the indices of the d largest eigenvalues
    idx = eigenValues.argsort()[::-1][:d]
    D = np.diag(eigenValues[idx]**p)
    W = eigenVectors[:, idx]

    return W, D