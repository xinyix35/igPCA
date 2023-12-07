"""
Rank Selection for double-structured Data
"""
# Authors: Xinyi Xie <xinyix35@uw.edu>

import numpy as np
from numpy import linalg as LA
from .GMD import GMD
import pandas as pd
import matplotlib.pyplot as plt


class SelectionMethod:
    '''
    This class is the parent of the classes of two rank selection models, that is, 
    GMD regression model and Posterior Mean Model

    Parameters
    ----------
    by_insert: boolean, indicating how the class is initialized
        if True, then all the components are inserted;
        else, input the following paramters to initialize

    X : {array-like, sparse matrix} of shape (int n, int p)

    Sigma : {array-like, sparse matrix} of shape (int n, int n)

    Phi : {array-like, sparse matrix} of shape (int p, int p)

    m: int, number of rows of the targeted block;
        if greater than 0, each blocks are partitioned when initializing
        otherwise, each block need to be inserted 

    t: int, number of columns of the targeted block;
        if greater than 0, each blocks are partitioned when initializing
        otherwise, each block need to be inserted 

    '''

    def __init__(self, by_insert=True, A=None, B=None, C=None, D=None,
                 Sigma_11=None, Sigma_12=None, Sigma_21=None, Sigma_22=None,
                 Phi_11=None, Phi_12=None, Phi_21=None, Phi_22=None,
                 X=None, Sigma=None, Phi=None, m=0, t=0):
        if by_insert == True:
            self.__init_data_by_input__(A, B, C, D)
            self.__init_sigma_by_input__(
                Sigma_11, Sigma_12, Sigma_21, Sigma_22)
            self.__init_phi_by_input__(Phi_11, Phi_12, Phi_21, Phi_22)
        else:
            self.X = X
            self.Sigma = Sigma
            self.Phi = Phi
            self.A, self.B, self.C, self.D = self.__partition__(
                self.data, m, t)
            self.Sigma_11, self.Sigma_12, self.Sigma_21, self.Sigma_22 = self.__partition__(
                self.Sigma, m, m)
            self.Phi_11, self.Phi_12, self.Phi_21, self.Phi_22 = self.__partition__(
                self.Phi, t, t)

    def __partition__(self, matrix, nrow, ncol):
        A = matrix[:nrow, :ncol]
        B = matrix[:nrow, ncol:]
        C = matrix[nrow:, :ncol]
        D = matrix[nrow:, ncol:]
        return tuple((A, B, C, D))

    def __init_data_by_input__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def __init_sigma_by_input__(self, A, B, C, D):
        self.Sigma_11 = A
        self.Sigma_12 = B
        self.Sigma_21 = C
        self.Sigma_22 = D

    def __init_phi_by_input__(self, A, B, C, D):
        self.Phi_11 = A
        self.Phi_12 = B
        self.Phi_21 = C
        self.Phi_22 = D

    def __compute_QH_norm__(self, H, Q, A):
        # compute the Q-H norm of matrix A
        numeric_1 = np.matmul(H, A)
        numeric_2 = np.matmul(Q, np.transpose(A))
        result = np.trace(np.matmul(numeric_1, numeric_2))
        return np.sqrt(result)

    def __mse_evaluation__(self, est_A):
        m, t = np.shape(est_A)
        error = self.A - est_A
        norm = self.__compute_QH_norm__(
            LA.inv(self.Sigma_11), LA.inv(self.Phi_11), error)
        result = norm / np.sqrt(m * t)
        return result


class PosteriorMean(SelectionMethod):
    '''
    This class computes the error of a pre-specified partition of data 
    at any rank k, under the Posterior Mean Model
    '''

    def __init__(self, by_insert, A=None, B=None, C=None, D=None,
                 Sigma_11=None, Sigma_12=None, Sigma_21=None, Sigma_22=None,
                 Phi_11=None, Phi_12=None, Phi_21=None, Phi_22=None,
                 X=None, Sigma=None, Phi=None, m=0, t=0):
        super().__init__(by_insert, A, B, C, D,
                         Sigma_11, Sigma_12, Sigma_21, Sigma_22,
                         Phi_11, Phi_12, Phi_21, Phi_22,
                         X, Sigma, Phi, m=m, t=t)

    def __fit_once__(self, data, H, Q, k):
        model = GMD(data, H, Q, k)
        model.fit()
        return model.X_hat

    def forward_fixed_rank(self, k):
        '''
        Apply GMD to B, C, D to obtained estmated means, then estimate 
        M_A by self-consistency theorem

        Paramater
        ---------
        k: int, must be greater than 0
            the specified rank that we computes the error

        Returns
        -------
        Etsimated means of M_A, M_B, M_C, and M_D
        '''
        # fit B
        H_B = LA.inv(self.Sigma_11)
        Q_B = LA.inv(self.Phi_22)
        est_B = self.__fit_once__(self.B, H_B, Q_B, k)

        # fit C
        H_C = LA.inv(self.Sigma_22)
        Q_C = LA.inv(self.Phi_11)
        est_C = self.__fit_once__(self.C, H_C, Q_C, k)

        # fit D
        est_D = self.__fit_once__(self.D, H_C, Q_B, k)

        # get the estimated value of M_A
        temp = np.matmul(est_B, LA.pinv(est_D))
        est_A = np.matmul(temp, est_C)
        return tuple((est_A, est_B, est_C, est_D))

    # compute the conditional mean of A by the estimated means for a fixed rank k
    def __compute_conditional_mean__(self, k):
        '''
        Compute the conditional mean of A by the 
        estimated means for a fixed rank k

        Paramater
        ---------
        k: int, must be greater than 0
            the specified rank that we computes the error

        Returns
        -------
        Posterior Mean of M_A
        '''
        est_A, est_B, est_C, est_D = self.forward_fixed_rank(k)
        inv_1 = np.matmul(self.Sigma_12, LA.inv(self.Sigma_22))
        inv_2 = np.matmul(LA.inv(self.Phi_22), self.Phi_21)
        mean_1 = np.matmul(inv_1, self.C - est_C)
        mean_22 = np.matmul(inv_1, self.D - est_D)
        mean_2 = np.matmul(self.B - est_B - mean_22, inv_2)
        result = est_A + mean_1 + mean_2
        return result

    def error_k(self, k):
        '''
        Paramater
        ---------
        k: int, must be greater than 0
            the specified rank that we computes the error

        Returns
        -------
        float: the error of estimation at rank k
        '''
        result = self.__compute_conditional_mean__(k)
        error = self.__mse_evaluation__(result)
        return error


class GMDRegression(SelectionMethod):
    '''
    This class computes the error of a pre-specified partition of data 
    at any rank k, under the GMD Regression Model
    '''

    def __init__(self, by_insert=True, A=None, B=None, C=None, D=None,
                 Sigma_11=None, Sigma_12=None, Sigma_21=None, Sigma_22=None,
                 Phi_11=None, Phi_12=None, Phi_21=None, Phi_22=None,
                 X=None, Sigma=None, Phi=None, m=0, t=0):
        super().__init__(by_insert, A, B, C, D,
                         Sigma_11, Sigma_12, Sigma_21, Sigma_22,
                         Phi_11, Phi_12, Phi_21, Phi_22,
                         X, Sigma, Phi, m=0, t=0)

    def __regression_fixed_rank__(self, k):
        H_D = LA.inv(self.Sigma_22)
        Q_D = LA.inv(self.Phi_22)
        # fit D
        model = GMD(self.D, H_D, Q_D, k)
        model.fit()
        # compute estimated value of beta
        result = Q_D @ model.V @  LA.inv(np.diag(model.D)
                                         ) @ np.transpose(model.U) @ H_D @ self.C
        return result

    def __estimate_A__(self, beta):
        error = self.C - np.matmul(self.D, beta)
        component_1 = np.matmul(self.B, beta)
        component_2 = np.matmul(self.Sigma_12, LA.inv(self.Sigma_22))
        component_2 = np.matmul(component_2, error)
        result = component_1 + component_2
        return result

    def error_k(self, k):
        '''
        Paramater
        ---------
        k: int, must be greater than 0
            the specified rank that we computes the error

        Returns
        -------
        float: the error of estimation at rank k
        '''
        beta = self.__regression_fixed_rank__(k)
        est_A = self.__estimate_A__(beta)
        error = self.__mse_evaluation__(est_A)
        return error


class spilt:
    '''
    This class motivates the data spilting process used in Bi-Cross Validation (bcv.py)
    by the following steps:
    1. Generating the partitions of rows and column with particular matrix size(n,p) and fold-size(h,l)
    2. Reorganize the (i,j)-block of matrix to the top left corner
    3. Generate the corresponding row/column indices for matrices after reorganizing

    Parameters
    ----------
    n: int, nodefault value, sample size of the original data

    p: int, no default value, number of covariates of the original data

    h: int, default = 5, number of partition(fold) in sample

    l: int, default = 5, number of partition(fold) in covariate

    Example
    ----------
    1. first define a split object, and process the data
    sp = _spilt(n,p,h,l)
    sp.forward()
    2. To obatin the blocks of data X when the (i,j)-block is moved to the top left corner 
    by calling the function split_data() in the class:
    A, B, C, D = sp.split_data(X, i, j)

    '''

    def __init__(self, n, p, h=5, l=5):
        # size of original data
        self.n = n
        self.p = p
        # numbers of partitons in row and column
        self.rowpar = h
        self.colpar = l
        # initialize row index and column index
        self.row_index = []
        self.row_size = []
        self.col_index = []
        self.col_size = []

    def __row_par_index__(self):
        """
        Obatin the size of each partition (in row), and thus obatin
        the next one index of each partition

        Example
        ----------
        row = [i for i in range(11)]
        h = 3
        row_size = [4,4,3]
        row_index = [0,4,8,11]

        To obtain the first partition, we have row[0:4] = [0,1,2,3]
        To obatin the second partition, we have row[4:8] = [4,5,6,7]
        To obatin the third partition, we have row[8:11] = [8,9,10]
        """
        m, r = np.divmod(self.n, self.rowpar)
        self.row_size = [m + 1] * r + [m] * (self.rowpar - r)
        self.row_index = [sum(self.row_size[:i])
                          for i in range(self.rowpar + 1)]

    def __col_par_index__(self):
        """
        generate the size of each partition (in column), and thus obatin
        the first index of each partition
        """
        m, r = np.divmod(self.p, self.colpar)
        self.col_size = [m + 1] * r + [m] * (self.colpar - r)
        self.col_index = [sum(self.col_size[:i])
                          for i in range(self.colpar + 1)]

    def forward(self):
        self.__row_par_index__()
        self.__col_par_index__()

    def __A_index__(self, i, j):
        """
        Paramater
        ---------
        i: int, index of row partition, starts from 0
        j: int, index of row partition, starts from 0

        Returns
        -------
        the row indices and column indices of the (i,j)-block of data
        """
        full_row = [t for t in range(self.n)]
        row = full_row[self.row_index[i]: self.row_index[i + 1]]
        full_col = [t for t in range(self.p)]
        col = full_col[self.col_index[j]: self.col_index[j + 1]]
        return tuple((row, col))

    def __B_index__(self, i, j):
        """
        Paramater
        ---------
        i: int, index of row partition, starts from 0
        j: int, index of row partition, starts from 0

        Returns
        -------
        the row indices and column indices of the (i,\j)-block of data
        """
        full_row = [t for t in range(self.n)]
        row = full_row[self.row_index[i]: self.row_index[i + 1]]
        if j == 0:
            col = list(range(self.col_index[j + 1], self.p))
        else:
            col = (
                list(range(self.col_size[j], self.col_index[j]))
                + list(range(self.col_size[j]))
                + list(range(self.col_index[j + 1], self.p))
            )
        return tuple((row, col))

    def __C_index__(self, i, j):
        """
        Paramater
        ---------
        i: int, index of row partition, starts from 0
        j: int, index of row partition, starts from 0

        Returns
        -------
        the row indices and column indices of the (\i,j)-block of data
        """
        if i == 0:
            row = list(range(self.row_index[i + 1], self.n))
        else:
            row = (
                list(range(self.row_size[i], self.row_index[i]))
                + list(range(self.row_size[i]))
                + list(range(self.row_index[i + 1], self.n))
            )
        full_col = [t for t in range(self.p)]
        col = full_col[self.col_index[j]: self.col_index[j + 1]]
        return tuple((row, col))

    def __D_index__(self, i, j):
        """
        Paramater
        ---------
        i: int, index of row partition, starts from 0
        j: int, index of row partition, starts from 0

        Returns
        -------
        the row indices and column indices of the (\i,\j)-block of data
        """
        if i == 0:
            row = list(range(self.row_index[i + 1], self.n))
        else:
            row = (
                list(range(self.row_size[i], self.row_index[i]))
                + list(range(self.row_size[i]))
                + list(range(self.row_index[i + 1], self.n))
            )
        if j == 0:
            col = list(range(self.col_index[j + 1], self.p))
        else:
            col = (
                list(range(self.col_size[j], self.col_index[j]))
                + list(range(self.col_size[j]))
                + list(range(self.col_index[j + 1], self.p))
            )
        return tuple((row, col))

    def __sigma_11_index__(self, i):
        """
        Paramater
        ---------
        i: int, index of row partition, starts from 0

        Returns
        -------
        the index list of the row covariance matrix for (i,*)-block
        """
        # full_row = [t for t in range(self.n)]
        # row = full_row[self.row_index[i]: self.row_index[i + 1]]
        return list(range(self.row_index[i], self.row_index[i + 1]))

    def __sigma_22_index__(self, i):
        """
        Paramater
        ---------
        i: int, index of row partition, starts from 0

        Returns
        -------
        the index list of the row covariance matrix for (\i,*)-block
        """
        if i == 0:
            return list(range(self.row_index[i + 1], self.n))
        else:
            return (
                list(range(self.row_size[i], self.row_index[i]))
                + list(range(self.row_size[i]))
                + list(range(self.row_index[i + 1], self.n))
            )

    def __sigma_12_index__(self, i):
        """
        Paramater
        ---------
        i: int, index of row partition, starts from 0

        Returns
        -------
        tthe row indices and column indices for the row covariance matrix between
        (i,*)-block and (\i,*)-block
        """
        r = list(range(self.row_index[i], self.row_index[i + 1]))
        if i == 0:
            c = list(range(self.row_index[i + 1], self.n))
        else:
            c = (
                list(range(self.row_size[i], self.row_index[i]))
                + list(range(self.row_size[i]))
                + list(range(self.row_index[i + 1], self.n))
            )
        return tuple((r, c))

    def __sigma_21_index__(self, i):
        """
        Paramater
        ---------
        i: int, index of row partition, starts from 0

        Returns
        -------
        tthe row indices and column indices for the row covariance matrix between
        (\i,*)-block and (i,*)-block
        """
        c, r = self.__sigma_12_index__(i)
        return tuple((r, c))

    def __phi_11_index__(self, j):
        """
        Paramater
        ---------
        j: int, index of column partition, starts from 0

        Returns
        -------
        the index list of the row covariance matrix for (*,j)-block
        """
        return list(range(self.col_index[j], self.col_index[j + 1]))

    def __phi_22_index__(self, j):
        """
        Paramater
        ---------
        j: int, index of column partition, starts from 0

        Returns
        -------
        the index list of the row covariance matrix for (*,\j)-block
        """
        if j == 0:
            return list(range(self.col_index[j + 1], self.p))
        else:
            return (
                list(range(self.col_size[j], self.col_index[j]))
                + list(range(self.col_size[j]))
                + list(range(self.col_index[j + 1], self.p))
            )

    def __phi_12_index__(self, j):
        """
        Paramater
        ---------
        i: int, index of row partition, starts from 0

        Returns
        -------
        tthe row indices and column indices for the column covariance matrix between
        (*,j)-block and (*,\j)-block
        """
        r = list(range(self.col_index[j], self.col_index[j + 1]))
        if j == 0:
            c = list(range(self.col_index[j + 1], self.p))
        else:
            c = (
                list(range(self.col_size[j], self.col_index[j]))
                + list(range(self.col_size[j]))
                + list(range(self.col_index[j + 1], self.p))
            )
        return tuple((r, c))

    def __phi_21_index__(self, j):
        """
        Paramater
        ---------
        i: int, index of row partition, starts from 0

        Returns
        -------
        tthe row indices and column indices for the column covariance matrix between
        (*,\j)-block and (*,j)-block
        """
        c, r = self.__phi_12_index__(j)
        return tuple((r, c))

    def split_data(self, X, i, j):
        """
        Paramater
        ---------
        X : {array-like, sparse matrix} of shape (int n, int p)

        i: int, index of row partition, starts from 0

        j: int, index of column partition, starts from 0

        Returns
        -------
        List of the four new blocks when (i,j)-block being the targeted partition
        """
        # New A, New B, New C, New D
        r, c = self.__A_index__(i, j)
        A = X[r][:, c]
        r, c = self.__B_index__(i, j)
        B = X[r][:, c]
        r, c = self.__C_index__(i, j)
        C = X[r][:, c]
        r, c = self.__D_index__(i, j)
        D = X[r][:, c]
        return tuple((A, B, C, D))

    def split_sigma(self, Sigma, i):
        """
        Paramater
        ---------
        Sigma : {array-like, sparse matrix} of shape (int n, int n)

        i: int, index of row partition, starts from 0


        Returns
        -------
        List of the four row covariance matrices when (i,j)-block being the targeted partition
        """
        r = self.__sigma_11_index__(i)
        Sigma_11 = Sigma[r][:, r]
        c = self.__sigma_22_index__(i)
        Sigma_22 = Sigma[c][:, c]
        Sigma_12 = Sigma[r][:, c]
        Sigma_21 = Sigma[c][:, r]
        return tuple((Sigma_11, Sigma_12, Sigma_21, Sigma_22))

    def split_phi(self, Phi, j):
        """
        Paramater
        ---------
        Phi : {array-like, sparse matrix} of shape (int p, int p)

        j: int, index of column partition, starts from 0


        Returns
        -------
        List of the four column covariance matrices when (i,j)-block being the targeted partition
        """
        r = self.__phi_11_index__(j)
        Phi_11 = Phi[r][:, r]
        c = self.__phi_22_index__(j)
        Phi_22 = Phi[c][:, c]
        Phi_12 = Phi[r][:, c]
        Phi_21 = Phi[c][:, r]
        return tuple((Phi_11, Phi_12, Phi_21, Phi_22))


class BiCrossValidation:
    """
    BCV is the class to carry out the bi-cross validation process


    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (int n, int p)

    Sigma : {array-like, sparse matrix} of shape (int n, int n)

    Phi : {array-like, sparse matrix} of shape (int p, int p)

    h: int, default = 5, number of partition(fold) in sample

    l: int, default = 5, number of partition(fold) in covariate

    K: {list(int), int}, candidiates for the rank of tiplet (X, H, Q)
        If K is an integer, the candidates will be [1, 2, ....K]

    method: str in {'pst', 'reg'}, default = 'pst'
    """

    def __init__(self, X, Sigma, Phi, h, l, K, method='pst'):
        self.X = X
        self.Sigma = Sigma
        self.Phi = Phi
        self.h = h
        self.l = l
        self.n, self.p = np.shape(self.X)
        self.block_error = np.zeros((h, l, len(K)))
        self.error_mean = np.zeros(len(K))
        # initialize spilt class
        self.sp = spilt(self.n, self.p, self.h, self.l)
        self.sp.forward()
        # initialize method, default method is posterior mean method
        self.method = method
        # initialize the rank selection result
        self.rank_min = -1
        self.rank_one_std = -1
        if isinstance(K, int):
            self.K = np.range(1, K+1)
        else:
            self.K = K
        # initialize standard error for each rank
        self.std_each_rank = [-1]*len(self.K)

    def _init_model_(self, i, j):
        A, B, C, D = self.sp.split_data(self.X, i, j)
        sigma_11, sigma_12, sigma_21, sigma_22 = self.sp.split_sigma(
            self.Sigma, i
        )
        # sigma_11, sigma_12, sigma_21, sigma_22 = self.sp.split_sigma(
        #     self.Sigma, j
        # )
        phi_11, phi_12, phi_21, phi_22 = self.sp.split_phi(self.Phi, j)
        if self.method == "pst":
            model = PosteriorMean(by_insert=True, A=A, B=B, C=C, D=D,
                                  Sigma_11=sigma_11, Sigma_12=sigma_12, Sigma_21=sigma_21, Sigma_22=sigma_22,
                                  Phi_11=phi_11, Phi_12=phi_12, Phi_21=phi_21, Phi_22=phi_22)
            return (model)
        elif self.method == "reg":
            model = GMDRegression(by_insert=True, A=A, B=B, C=C, D=D,
                                  Sigma_11=sigma_11, Sigma_12=sigma_12, Sigma_21=sigma_21, Sigma_22=sigma_22,
                                  Phi_11=phi_11, Phi_12=phi_12, Phi_21=phi_21, Phi_22=phi_22)
            return (model)
        else:
            print("Wrong Method Input")
            return -1

    def fit(self):
        """
        Fit the Rank Selection model with different rank for all the h*l folds,
        and store the result in an array self.error_mean:
            store mean error for every fold, with $K$ different values corresponding 
            to different ranks in K(list)
        """
        for i in range(self.h):
            for j in range(self.l):
                select_model = self._init_model_(i, j)
                """fit the models with different rank"""
                error = []
                for k in self.K:
                    error.append(select_model.error_k(k))
                self.block_error[i, j, :] = error
        self.error_mean = np.mean(self.block_error, axis=0)
        self.error_mean = np.mean(self.error_mean, axis=0)

    def rank_selection(self, plot=True, fig_name=None):
        """
        Returns
        --------
        ranks by both minimum error as well as one-stanard error rule
        """
        col_name = [str(k) for k in self.K]
        df = pd.DataFrame(self.block_error.reshape(-1,
                          len(self.K)), columns=col_name)
        """one-std rule"""
        min_index = np.argmin(self.error_mean)
        self.rank_min = self.K[min_index]
        std_k = df.apply(np.std, axis=0).values
        std_k = std_k / np.sqrt(self.h * self.l)
        min_val_std = std_k[min_index]
        min_error = np.min(self.error_mean)
        range_error_id = [
            index
            for index in range(len(self.K))
            if (
                self.error_mean[index] >= min_error - min_val_std
                and self.error_mean[index] <= min_error + min_val_std
            )
        ]
        self.rank_one_std = self.K[np.min(range_error_id)]
        if plot:
            plt.errorbar(x=self.K, y=self.error_mean, yerr=std_k)
            plt.savefig(fig_name+".png")
        return tuple((self.rank_one_std, self.rank_min))
