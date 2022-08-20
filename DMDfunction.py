# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 20:24:21 2022

@author: brown
"""

from numpy import reciprocal, diag
from numpy.linalg import svd, eig, pinv

def DMD(data, r):
    """Dynamic Mode Decomposition (DMD) algorithm."""
    
    ## Build data matrices
    X1 = data[:, : -1]
    X2 = data[:, 1 :]
    ## Perform singular value decomposition on X1
    u, s, v = svd(X1, full_matrices = False)
    ## Compute the Koopman matrix
    A_tilde = u[:, : r].conj().T @ X2 @ v[: r, :].conj().T * \
            reciprocal(s[: r])
    ## Perform eigenvalue decomposition on A_tilde
    Phi, Q = eig(A_tilde)
    ## Compute the coefficient matrix
    Psi = X2 @ v[: r, :].conj().T @ diag(reciprocal(s[: r])) @ Q
    A = Psi @ diag(Phi) @ pinv(Psi)
    
    return A_tilde, Phi, A

import numpy as np
def DMD4cast(data, r, pred_step):
    N, T = data.shape
    _, _, A = DMD(data, r)
    mat = np.append(data, np.zeros((N, pred_step)), axis = 1)
    for s in range(pred_step):
        mat[:, T + s] = (A @ mat[:, T + s - 1]).real
    return mat[:, - pred_step :]

X = np.zeros((2, 10))
X[0, :] = np.arange(1, 11)
X[1, :] = np.arange(2, 12)
pred_step = 2
pred_step=4
r = 2
mat_hat = DMD4cast(X, r, pred_step)
print(mat_hat)