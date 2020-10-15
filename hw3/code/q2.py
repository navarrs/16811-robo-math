# @author     Ingrid Navarro
# @andrewid   ingridn
# @date       Oct 10, 2020
#
# @brief      Hw2 - Q2 Data fitting

#
# Includes ---------------------------------------------------------------------
import argparse
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


def GeneratePoly(x, i):
    if i == 0:
        return x**i, f"1"
    return x**i, f"x**{i}"


def GenerateSin(x, n):
    return np.sin(n*np.pi*x), f"sin({n}*pi*x)"


def GenerateCos(x, n):
    return np.cons(n*np.pi*x), f"cos({n}*pi*x)"


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--npoly", default=2, type=int,
                        help="Number of polynomial basis function")
    parser.add_argument("--p", default=0, type=int, 
                        help="Starting value of i in x**i")
    parser.add_argument("--nsin", default=1, type=int,
                        help="Number sin() basis functions")
    parser.add_argument("--s", default=5, type=int, 
                        help="Starting value of n in sin(n*pix)")
    parser.add_argument("--ncos", default=0, type=int,
                        help="Number cos() basis functions")
    parser.add_argument("--c", default=5, type=int, 
                        help="Starting value of n in cos(n*pix)")
    args = parser.parse_args()
    
    inc = 0.01
    y = np.loadtxt("../data/problem2.txt", dtype=np.float)
    x = np.arange(start=0, stop=1+inc, step=inc)

    # Create matrix A with basis function
    m = args.npoly + args.nsin + args.ncos
    A = np.zeros((x.shape[0], m), dtype=np.float)
    
    f_str = ""
    # Add Polynomials
    p = args.p
    for i in range(args.npoly):
        A[:, i], s_ = GeneratePoly(x, p)
        f_str += s_ + " + "
        p += 1
        
    # Generate Sin
    s = args.s
    for i in range(args.nsin):
        A[:, i+args.npoly], s_ = GenerateSin(x, s)
        f_str += s_ + " + "
        s += 1
        
    # Generate Cos
    c = args.c 
    for i in range(args.ncos):
        A[:, i+args.npoly+args.nsin], s_ = GenerateCos(x, c)
        f_str += s_ + " + "
        c += 1
    
    y_, c = Approximate(A, y)
    print(f"--------------------------------------------")
    print(f"Coefficients: {c}")
    title = f"Recontstruction f(x) = {f_str}"
    print(title)
    Plot(x, y, x, y_, title)
