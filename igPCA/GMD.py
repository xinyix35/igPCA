"""Sparse Generalized Matrix Decomposition"""

# Authors: Xinyi Xie <xinyix35@uw.edu>
#           Jing Ma <jingma@fredhutch.org>
# R version: https://github.com/drjingma/GMDecomp

import numpy as np
from numpy import linalg as LA


class GMD:
    """
    This class implements the Generalized Matrix Decomposition (GMD) and supports sparse loading and score given by lasso.

    Parameters
    ----------
    X : {array-like, sparse matrix of shape (int n, int p)}

    H : {array-like, matrix} of shape (int n, int n)
        Matrix Characterizing the (dis)similarity structure of sample space of X

    Q : {array-like, matrix} of shape (int p, int p)
        Matrix Characterizing the (dis)similarity structure of variable space of X

    K : int, default=3
        Number of components to keep

    max_iter : int, default=50
        The maximum number of iterations.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are smaller than tol, the 
        optimization code checks the dual gap for optimality and continues until it 
        is smaller than tol

    lu : float, default=None
        Constant that multiplies the L1 term with respect to score(U), 
        controlling regularization strength. lu must be a non-negative float i.e. in [0, inf).

    lv : float, default=None
        Constant that multiplies the L1 term with respect to loading(V), 
        controlling regularization strength. lu must be a non-negative float i.e. in [0, inf).


    Attributes
    ----------
    U : {array-like, matrix} of shape (n, K)
        Estimated GMD score 

    D : {array-like} of shape (K)
        Estimated GMD value 

    V : {array-like, matrix} of shape (p, K)
        Estimated GMD loading

    X_hat : {array-like, matrix} of shape (n, K)
        Estimated GMD value, it is equivalent to the mean matrix of X when assuming 
        ``X ~ MN_{n,p}(X_hat, H^{-1}, Q^{-1})``
    """

    def __init__(self, X, H, Q, K=3, max_iter=50, tol=1e-4):
        self.n, self.p = X.shape
        # to store the original data
        self.X = X
        self.H = H
        self.Q = Q
        # k is the rank for GMD specified in advanced
        self.K = K
        # initialize parameters for iterations
        self.u = np.zeros(self.n).reshape(self.n, 1)
        self.v = np.zeros(self.p).reshape(self.p, 1)
        self.d = np.zeros(1)
        # initialze paramters for controlling iterations
        self.max_iter = max_iter
        self.tol = tol
        # initialize parameters for output
        self.U = np.zeros(self.n * self.K).reshape(self.n, self.K)
        self.V = np.zeros(self.p * self.K).reshape(self.p, self.K)
        self.D = np.zeros(self.K)

    def __soft_thresholding__(self, l, x):
        p = np.shape(x)[0]
        return np.sign(x)*np.maximum(np.zeros((p, 1)), np.abs(x)-l)

    def __compute_A_norm__(self, A, vec):
        # A is the matirx specifying the normed-space with size n * p
        # vec is a vector with shape p * 1, p is any natural number
        # norm is given as (v^t)Av
        result = np.matmul(np.transpose(vec), A)
        result = np.matmul(result, vec)
        return np.sqrt(result)

    # initialize u and v at each rank $k = 1, ..., K$ so that
    # u^T H u >0 and  v^T Q v >0
    def __initialize_uv__(self, component):
        self.u = np.zeros(self.n).reshape(self.n, 1)
        self.v = np.zeros(self.p).reshape(self.p, 1)
        self.u[0] = 1
        self.v[0] = 1
        if component >= (self.n - 1):
            self.u[0, 0] = 1
        else:
            i = 1
            self.u[component, 0] = 1
            while True:
                temp = np.matmul(np.transpose(self.u), self.H)
                temp = np.matmul(temp, self.u)
                if temp < 0:
                    self.u[component + i, 0] = 1
                    i = i + 1
                else:
                    break
        self.v = np.zeros(self.p).reshape(self.p, 1)
        if component > (self.p - 1):
            self.v[0, 0] = 1
        else:
            j = 1
            self.v[component, 0] = 1
            while True:
                temp = np.matmul(np.transpose(self.v), self.Q)
                temp = np.matmul(temp, self.v)
                if temp < 0:
                    self.v[component + j, 0] = 1
                    j = j + 1
                else:
                    break

    # update u and v for once
    def __compute_uv_vector__(self, lu, lv):
        cons_u = np.matmul(self.X, self.Q)
        cons_v = np.matmul(np.transpose(self.X), self.H)
        # compute u
        u_nscale = self.__soft_thresholding__(
            lu, np.matmul(cons_u, self.v))
        u_norm = self.__compute_A_norm__(self.H, u_nscale)
        self.u = (u_norm > 0) * u_nscale / u_norm
        # compute v
        v_nscale = self.__soft_thresholding__(
            lv, np.matmul(cons_v, self.u))
        v_norm = self.__compute_A_norm__(self.Q, v_nscale)
        self.v = (v_norm > 0) * v_nscale / v_norm

    def __compute_d__(self):
        temp = np.matmul(np.transpose(self.u), self.H)
        temp = np.matmul(temp, self.X)
        temp = np.matmul(temp, self.Q)
        self.d = np.matmul(temp, self.v)

    def __compute_X__(self):
        new_mass = self.d * np.matmul(self.u, np.transpose(self.v))
        self.X = self.X - new_mass

    def __GMD_result__(self):
        # the result is given as UDV^t
        component = np.matmul(self.U, np.diag(self.D))
        fitted_value = np.matmul(component, np.transpose(self.V))
        return fitted_value

    def fit(self, lu=0, lv=0):
        # iterates over rank
        for component in range(self.K):
            # update u,v
            self.__initialize_uv__(component)
            # u_0 and v_0 serves as a placeholder for results in previous iteration
            u0 = self.u
            v0 = self.v
            error = 1
            num_iter = 1
            # keep computing till convergence or reach the maximum iterations
            while error > self.tol:
                # compute u and v
                self.__compute_uv_vector__(lu, lv)
                num_iter = num_iter + 1
                # validate whether the algorithm converges
                error = LA.norm(self.u - u0, 2) + LA.norm(self.v - v0, 2)
                # prepare for the next iteration
                u0 = self.u
                v0 = self.v
                if num_iter > self.max_iter:
                    break
            # u_k and v_k are determined, update the output matrices U and V
            self.U[:, component] = self.u.reshape(self.n)
            self.V[:, component] = self.v.reshape(self.p)
            # update d via the given u and v
            self.__compute_d__()
            self.D[component] = self.d
            # update X
            self.__compute_X__()
        self.X_hat = self.__GMD_result__()
