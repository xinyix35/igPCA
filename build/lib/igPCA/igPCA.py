"""Integrative Generalized Principle Component Analysis"""

# Authors: Xinyi Xie <xinyix35@uw.edu>

import numpy as np
from numpy import linalg as LA
from .GMD import GMD
from .Selection import BiCrossValidation


class igPCA:
    """ 
    This class implements the Integrative Generalized Principle Component Analysis(igPCS)
    and supports sparse loading and score given by lasso.

    Parameters
    ----------
    X1 : array-like matrix of shape (int n, int p1)

    X2 : array-like matrix of shape (int n, int p2)

    H : array-like, matrix of shape (int n, int n)
        Matrix Characterizing the (dis)similarity structure of sample space of X1 and X2

    Q1 : array-like, matrix of shape (int p1, int p1)
        Matrix Characterizing the (dis)similarity structure of variable space of X1

    Q2 : array-like, matrix of shape (int p2, int p1)
        Matrix Characterizing the (dis)similarity structure of variable space of X2

    r1: int, the total rank of X1; Defaults to None.

    r2: int, the total rank of X2; Defaults to None.

    thres: float bewtween (0,1), threshold for select the joint rank
        Defaults to 0.9.
    """

    def __init__(self, X1, X2, H, Q1, Q2, r1=None, r2=None, thres=0.9):
        self.X1 = X1
        self.X2 = X2
        self.H = H
        self.Q1 = Q1
        self.Q2 = Q2
        self.r1 = r1
        self.r2 = r2
        self.thres = thres

    def __rank_selection__(self, X, sigma, phi, K, h, l, method):
        bcv_class = BiCrossValidation(X, sigma, phi, h, l, K, method)
        # forward and process the rank selection
        bcv_class.fit()
        rank_std, rank_min = bcv_class.rank_selection(plot=False)
        return tuple((rank_std, rank_min, bcv_class.error_mean))

    # modification required: h, l
    def X1_rank_selection(self, K=None, method='pst', h=10, l=10, std=True):
        """
        Selects the rank of X1 of r1 is unknown

        Parameters
        ----------
        K (list, optional): candidates of r1. Defaults to None.

        method (str, optional): Rank selection Method. Defaults to 'pst'.

        h (int, optional): Number of folds in row under BCV framework. Defaults to 10.

        l (int, optional): Number of folds in row under BCV framework. Defaults to 10.

        std (bool, optional): Indicating whether select the rank by one-standard deviation rule.
            Defaults to True.

        Modifies
        -------
        self.r1
        """
        rank_std, rank_min, self.rs_error_x1 = self.__rank_selection__(
            self.X1, self.sigma, self.phi_1, h, l, K, method)
        if std:
            self.r1 = rank_std
        else:
            self.r1 = rank_min
        print("rank selected for X1 with minimum error is " + str(rank_min))
        print("rank selected for X1 one-standard deviation rule is " + str(rank_std))

    # modification required: h, l
    def X2_rank_selection(self, K=None, method='pst', std=True):
        """
        Selects the rank of X1 of r1 is unknown

        Parameters
        ----------
        K (list, optional): candidates of r2. Defaults to None.

        method (str, optional): Rank selection Method. Defaults to 'pst'.

        h (int, optional): Number of folds in row under BCV framework. Defaults to 10.

        l (int, optional): Number of folds in row under BCV framework. Defaults to 10.

        std (bool, optional): Indicating whether select the rank by one-standard deviation rule.
            Defaults to True.

        Modifies
        -------
        self.r2
        """
        rank_std, rank_min, self.rs_error_x2 = self.__rank_selection__(
            self.X2, self.sigma, self.phi_2, 10, 10, K, method)
        if std:
            self.r2 = rank_std
        else:
            self.r2 = rank_min
        print("rank selected for X2 with minimum error is " + str(rank_min))
        print("rank selected for X2 one-standard deviation rule is " + str(rank_std))

    def __separate_GMD__(self, X, H, Q, rank):
        model = GMD(X=X, H=H, Q=Q, K=rank)
        model.fit()
        return_list = tuple((model.U, model.D, model.V, model.X_hat))
        return (return_list)

    # compute the inner product of two vectors/matrices under H-norm
    def __H_inner_prod__(self, u, v):
        a = np.matmul(np.transpose(u), self.H)
        a = np.matmul(a, v)
        return a

    def __joint_ingredient__(self):
        # Fit X1 and X2 by GMD separately
        U1,  _, _, X_1_tilde = self.__separate_GMD__(
            self.X1, self.H, self.Q1, self.r1)
        U2, _, _, X_2_tilde = self.__separate_GMD__(
            self.X2, self.H, self.Q2, self.r2)
        return tuple((U1, U2, X_1_tilde, X_2_tilde))

    def __joint_rank_evaluation__(self, U1, U2, threshold=None):
        if (threshold):
            self.threshold = threshold
        joint_prod = self.__H_inner_prod__(U1, U2)
        _, s, _ = LA.svd(joint_prod, full_matrices=False, compute_uv=True)
        self.r0 = sum(s > self.threshold)

    def __joint_fit__(self, U1, U2, X_1_tilde, X_2_tilde):
        U_joint = np.concatenate((U1, U2), axis=1)
        p_joint = np.shape(U_joint)[1]
        # Fit the joint part
        joint_gmd = GMD(U_joint, self.H, np.eye(p_joint), self.r0)
        joint_gmd.fit()
        self.U0 = joint_gmd.U
        L = joint_gmd.U
        proj = np.matmul(L, np.transpose(L))
        proj = np.matmul(proj, self.H)
        J1 = np.matmul(proj, X_1_tilde)
        J2 = np.matmul(proj, X_2_tilde)
        _, self.D01, self.V01, self.J1 = self.__separate_GMD__(
            J1, self.H, self.Q1, self.r0)
        _, self.D02, self.V02, self.J2 = self.__separate_GMD__(
            J2, self.H, self.Q2, self.r0)
        return tuple((J1, J2))

    def __individual_fit__(self, X_1_tilde, X_2_tilde, J1, J2):
        ind_1 = X_1_tilde - J1
        ind_2 = X_2_tilde - J2
        self.U1, self.D11, self.V11, self.A1 = self.__separate_GMD__(
            ind_1, self.H, self.Q1, self.r1-self.r0)
        self.U2, self.D12, self.V12, self.A2 = self.__separate_GMD__(
            ind_2, self.H, self.Q2, self.r2-self.r0)

    def fit(self, r0=None, rank_method='pst', K1=None, K2=None, thres=None, h1=None, l1=None, h2=None, l2=None):
        """
        This function implements the igPCA algo in the followsing steps:

        1. Rank selections of X1 and X2 if required
            Parameters
            ----------
            rank_method: str, optional. Rank selection method. Defaults to 'pst'

            K1: list, optional. Candidates of r1. Defaults to None.

            K2: list, optional. Candidates of r2. Defaults to None.

            h1, h2: int, optional. Number of folds in row under BCV framework. Defaults to 10.

            l1, l2: int, optional. Number of folds in row under BCV framework. Defaults to 10.

        2. Estimate joint rank by pre-specified or user-specified threshold
            Parameters
            ----------
            r0 : int, optional. Defaults to None.

        3. Estimate the joint componenets

        4. Estimate the joint componenets
        """
        if self.r1 == None:
            self.X1_rank_selection(K=K1, method=rank_method, h=h1, l=l1)
        if self.r2 == None:
            self.X2_rank_selection(K=K2, method=rank_method, h=h2, l=l2)
        U1,  U2, X_1_tilde, X_2_tilde = self.__joint_ingredient__()
        if (r0):
            self.r0 = r0
        else:
            self.__joint_rank_evaluation__(U1, U2, thres)
        J1, J2 = self.__joint_fit__(U1, U2, X_1_tilde, X_2_tilde)
        self.__individual_fit__(X_1_tilde, X_2_tilde, J1, J2)
        # complete estimation
        self.X_1_hat = self.J1 + self.A1
        self.X_2_hat = self.J2 + self.A2
