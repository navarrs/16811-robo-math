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
# Problem implementation -------------------------------------------------------
class SolutionType(Enum):
  UNIQUE_SOL = 1
  ZERO_SOL = 2
  MANY_SOL = 3
  
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
  sol_type = SolutionType.UNIQUE_SOL
  n = A.shape[0] # rows
  m = A.shape[1] # cols
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
  x  = np.dot(np.matmul(np.matmul(V, S_inv), U_T), b)
  ns = []

  # If b in colspace(A), then Ax = b with x = V 1/S U_T b
  # Check if the type of solution to the system
  if k < n:
    b_ = np.matmul(A, x)
    ns = V_T[-(n-k):, :]
    if np.allclose(b, b_):
      # b is in colspace(A) but there are many solutions
      sol_type = SolutionType.MANY_SOL
    else:
      # b is not in colspace(A)
      sol_type = SolutionType.ZERO_SOL

  return x, ns, sol_type

#
# Helper methods ---------------------------------------------------------------

def Plot(x, y, x2, y2, title='f(x)'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.plot(x, y, color='r', label='f(x)')
    plt.plot(x2, y2, color='b', label='f(x)')
    plt.legend(loc='upper left')
    plt.show()

def CreateBasis(x, n):
  basis = np.zeros(shape=(x.shape[0], n*2), dtype=np.float)
  basis[:, 0] = 1.0
  for i in range(1, n):
    basis[:, i] = x**i
  
  # for i in range(n, n*2):
  #   basis[:, i] = 
  
  return basis

def CreateBasisTrig(x, n):
  basis = np.zeros(shape=(x.shape[0], n), dtype=np.float)
  for i in range(n):
    if i % 2 == 0:
      basis[:, i] = np.cos((i+1)*np.pi*x)
    else:
      basis[:, i] = np.sin((i+1)*np.pi*x)
  return basis

#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":

    # Q1.B
    y = np.loadtxt("../data/problem2.txt", dtype=np.float)
    inc = 0.01
    x = np.arange(start=0, stop=1+inc, step=inc)
    
    # 1, x, x**2, cos(pix), sin(pix)
    basis = np.zeros(shape=(x.shape[0], 9), dtype=np.float)
    basis[:, 0] = 1
    basis[:, 1] = x
    basis[:, 2] = x**2
    basis[:, 3] = np.cos(np.pi*x)
    basis[:, 4] = np.sin(np.pi*x)
    basis[:, 5] = np.cos(2*np.pi*x)
    basis[:, 6] = np.sin(2*np.pi*x)
    basis[:, 7] = np.cos(3*np.pi*x)
    basis[:, 8] = np.sin(3*np.pi*x)
    
    # basis = CreateBasis(x, 2)
    
    # basis = CreateBasisTrig(x, 4)
    # print(basis)
    
    coeffs, ns, sol = Solve(basis, y)
    # print(coeffs, sol)
    
    y_ = np.matmul(basis, coeffs.T)
    # y_ = y.copy()
    # for i in range(y.shape[0]):
    #   y_[i] = coeffs[0] + coeffs[1]*x[i] + coeffs[2] * x[i] **2
    #   y_[i] += coeffs[3] * np.cos(np.pi*x[i]) + coeffs[4] * np.sin(np.pi*x[i])

    Plot(x, y, x, y_)
    
    
      
    
