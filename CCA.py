# -*- coding: utf-8 -*-
from CCA_utils import *

## Visual features
V = np.ones((1000, 500))
d1 = V.shape[1]
phi_V = mapVisualFeatures(V)

## Tag features
T = np.ones((1000, 400))
d2 = T.shape[1]
phi_T = mapTagFeatures(T)

S, S_D = computeCovMatrix([phi_V, phi_T])

d = 128
p = 4
regularization = 1e-4
W, D = solveCCA(S, S_D, d, regularization, p)

W1 = W[:d1, :]
W2 = W[d1:, :]