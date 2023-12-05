#!/usr/bin/env python
from scipy.linalg import subspace_angles
from igPCA import igPCA
from igPCA import igPCA
from numpy import linalg as LA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
np.set_printoptions(suppress=True)

sns.set_theme()


def square_error(x, y):
    return LA.norm(x-y)


def compute_QH_norm(A, H, Q):
    n, p = np.shape(A)
    numeric_1 = np.matmul(H, A)
    numeric_2 = np.matmul(Q, np.transpose(A))
    result = np.trace(np.matmul(numeric_1, numeric_2))
    return np.sqrt(result)/np.sqrt(n*p)


def principle_angle(A, B):
    angle = subspace_angles(A, B)
    return (1 - np.cos(angle[0]))


def baseline_PCA(X1, X2, k):
    joint_matrix = np.concatenate((X1, X2), axis=1)
    u, s, v = LA.svd(joint_matrix, full_matrices=False, compute_uv=True)
    x_pca = np.dot(u[:, :k]*s[:k], v[:k, :])
    return tuple((u[:, :k], x_pca))


def sym_eig(X):
    val, vec = LA.eig(X)
    sQ = vec @ np.diag(np.sqrt(val)) @ vec.T
    isQ = vec @ np.diag(1/np.sqrt(val)) @ vec.T
    return (tuple((sQ, isQ)))


def coIA(X1, X2, H, Q1, Q2, K):
    sQ1, isQ1 = sym_eig(Q1)
    sQ2, isQ2 = sym_eig(Q2)
    w = sQ1.T @ X1.T @ H @ X2 @ sQ2
    ea, d, eb = LA.svd(w)
    A = ea[:, : K]
    B = eb.T[:, : K]
    V1 = isQ1 @ A
    V2 = isQ2 @ B
    U1 = X1 @ Q1 @ V1
    U2 = X2 @ Q2 @ V2
    return (tuple((U1[:, :K], V1[:, :K], U2[:, :K], V2[:, :K])))


def BIC(X, M_hat, V, H, Q1):
    n, p = np.shape(X)
    res = X - M_hat
    logloss = np.log(compute_QH_norm(res, H, Q1)**2/(n*p))
    sps = np.sum(V > 1e-5)
    complexity = sps * np.log(n*p)/(n*p)
    result = (logloss+complexity)
    return result


'''
Assumptions on the model input:
1. r0 is the joint rank, and r1 and r2 are total ranks(joint + individual) for X1 and X2
2.1 d0 is the joint signal, and d1 and d2 are total signals(joint + individual) for X1 and X2
2.2 signals d0, d1, d2 are sorted in the descending pattern within each block for convenience.
    Caution: it might not be sorted when combine altogether! Need sorting when combining altogether
3. In this simulation study, we assume that joint rank r_0 is known, and we do not process the 
    rank selection  procedure.
'''


def M_generation(U, V, d):
    s = len(d)
    M = np.dot(d * U[:, :s], V[:, :s].T)
    return M


def normalize_matrix(H):
    t = np.trace(H)
    n = np.shape(H)[0]
    H = H/t*n
    return H


def X_generation(H, Q, M, noise_level):
    n, p = np.shape(M)
    noise = np.random.normal(0, 1, (n, p))
    sigma = normalize_matrix(LA.inv(H))
    phi = normalize_matrix(LA.inv(Q))
    v1, p1 = LA.eig(sigma)
    v2, p2 = LA.eig(phi)
    L = np.matmul(np.matmul(p1, np.sqrt(np.diag(v1))), p1.T)
    R = np.matmul(np.matmul(p2, np.sqrt(np.diag(v2))), p2.T)
    noise = L @ noise @ R
    X = M + noise * noise_level
    return X


d01 = [100]
d02 = [80]
d11 = [50]
d12 = [40]
noise_1 = 0.8
noise_2 = 0.6
r0 = 1
r1 = 2
r2 = 2

# read data
U = np.loadtxt('../simulation_settings/U_0.8.csv', delimiter=",", dtype=float)
H = np.loadtxt('../simulation_settings/H_0.8.csv', delimiter=",", dtype=float)
V1 = np.loadtxt('../simulation_settings/V1_0.8.csv',
                delimiter=",", dtype=float)
V2 = np.loadtxt('../simulation_settings/V2_0.6.csv',
                delimiter=",", dtype=float)
Q1 = np.loadtxt('../simulation_settings/Q1_0.8.csv',
                delimiter=",", dtype=float)
Q2 = np.loadtxt('../simulation_settings/Q2_0.6.csv',
                delimiter=",", dtype=float)
U1 = U[:, [1, 0]]
U2 = U[:, [1, 2]]
M1 = M_generation(U1, V1, d01+d11)
M2 = M_generation(U2, V2, d02+d12)
X1 = X_generation(H, Q1, M1, noise_1)
X2 = X_generation(H, Q2, M2, noise_2)
print('Finish data generation')
n, p1 = np.shape(X1)
_, p2 = np.shape(X2)

# Data Prepatation
V_11 = V1[:, r0:]
V_12 = V2[:, r0:]
V_01 = V1[:, :r0]
V_02 = V2[:, :r0]
# just for this simulation
U0_true = U1[:, 0]
U1_true = U1[:, 1]
U2_true = U2[:, 1]
# use particular H, Q1
# # BASELINE 1: coIA
_, v1_coia, _, v2_coia = coIA(X1, X2, H, Q1, Q2, 2)
print('coIA Done')
# BASELINE 2: JIVE
model1 = igPCA(X1, X2, np.eye(n), np.eye(
    p1), np.eye(p2), r1=r1, r2=r2)
model1.fit(r0=r0, thres=0.9)
u0_jive = model1.U0
v11_jive = model1.V11
v12_jive = model1.V12
v01_jive = model1.V01
v02_jive = model1.V02
u1_jive = model1.U1
u2_jive = model1.U2
x1_jive = model1.X_1_hat
x2_jive = model1.X_2_hat
print('JIVE Done')
# Our Model 3: igPCA
model2 = igPCA(X1, X2, H, Q1, Q2, r1=r1, r2=r2)
model2.fit(r0=r0)
u0_igpca = model2.U0
v11_igpca = model2.V11
v12_igpca = model2.V12
v01_igpca = model2.V01
v02_igpca = model2.V02
u1_igpca = model2.U1
u2_igpca = model2.U2
x1_igpca = model2.X_1_hat
x2_igpca = model2.X_2_hat
print('igPCA Done')

plt.plot(U0_true, label='true')
plt.plot(u0_jive, label='jive')
plt.plot(u0_igpca, label='igpca')
plt.legend(loc='best')
plt.show()
