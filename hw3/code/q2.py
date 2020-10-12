# @author     Ingrid Navarro
# @andrewid   ingridn
# @date       Oct 10, 2020
#
# @brief      Hw2 - Q2 Data fitting

#
# Includes ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from scipy import linalg

#
# Helper methods ---------------------------------------------------------------

def Plot(x, y, x2, y2, title='Data fitting plot'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.scatter(x, y, color='m', label='Data', s=8)
    plt.plot(x2, y2, color='b', label='Fitted f(x)')
    plt.legend(loc='upper left')
    plt.show()


def CreateBasis(x, n_poly=1, n_fourier=1, n=1):
    basis = np.zeros(shape=(x.shape[0], n_poly+n_fourier), dtype=np.float)

    if n_poly >= 1:
        basis[:, 0] = 1.0

    # Add polynomial basis
    for i in range(1, n_poly):
        basis[:, i] = x**i

    for i in range(n_poly, n_poly+n_fourier):
        if i % 2 == 0:
            basis[:, i] = np.cos(n*np.pi*x)
        else:
            basis[:, i] = np.sin(n*np.pi*x)
        n += 1

    return basis

def Approximate(A, f):
  c = Solve(A, f)
  f_ = np.matmul(A, c.T)
  return f_, c

#
# From Homework 1
# ------------------------------------------------------------------------------

def ComputeSVD(A):
    """
      Computes singular value decomposition
      Input
      -----
        A: mxn matrix
      Output
      ------
        U: mxm matrix, change of basis for colspace(A)
        S: mxn matrix, eigenvalues of A.A_T, A_T.A
        V_T: nxn matrix, change of basis for rowspace(A) + nullspace(A)
    """
    m = A.shape[0]
    n = A.shape[1]
    U, s, V_T = linalg.svd(A)
    S = linalg.diagsvd(s, m, n)
    return U, S, V_T


def Solve(A, b):
    """
      Computes the least squares approach to find x through SVD
      -----
        A: mxn matrix
        b: 1xn vector
      Output
      ------
        x: The solution to the system of linear equations.
        ns: Vector(s) that span the null-space 
        sol_type: If the system has UNIQUE, ZERO or MANY solutions
    """
    n = A.shape[0]  # rows
    m = A.shape[1]  # cols
    # print(f"\nA^{n}x{m}:\n{A}\nb:\n{b}\n")

    U, S, V_T = ComputeSVD(A)
    k = np.linalg.matrix_rank(S)
    # print(f"\n* SVD Decomposition\nU:\n{U}\nS:\n{S}\nV_T:\n{V_T}")

    # This is solved through the least-norm approach seen in class
    U_T = np.transpose(U)
    S_inv = S
    for i in range(k):
        S_inv[i, i] = 1 / S_inv[i, i]
    S_inv = S_inv.transpose()
    V = np.transpose(V_T)

    # x = V 1/S U_T b
    x = np.dot(np.matmul(np.matmul(V, S_inv), U_T), b)
    return x


#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":

    # Q1.B
    y = np.loadtxt("../data/problem2.txt", dtype=np.float)
    inc = 0.01
    x = np.arange(start=0, stop=1+inc, step=inc)

    # 1, x, x**2, cos(pix), sin(pix)
    # 3 poly 8 fourier
    # basis = CreateBasis(x, 3, 8)
    
    # 3 poly 4 fourier cos1 sin2 cos3 sin4
    # A = CreateBasis(x, 3, 4)
    A = CreateBasis(x, 3, 4)
    y_, c = Approximate(A, y)
    print(f"Coefficients: {c}")
    Plot(x, y, x, y_)
    
