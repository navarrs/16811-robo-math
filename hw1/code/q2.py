# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 4, 2020
#
# @brief      Computing LDU and SVD decomposition 

#
# Includes ---------------------------------------------------------------------
import sys
import glob

import utils
import q1
import numpy as np
from scipy import linalg 

np.set_printoptions(precision=3, suppress=True)

#
# Functions --------------------------------------------------------------------
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

def ComputeLDU(A):
  """
    Performs LDU decomposition via Gaussian reduction.
    Input
    ------
      A: Square (mxn, n=m), non-singular matrix that will be decomposed
    Output
    ------
      L: Low triangular matrix (mxm)
      D: Diagonal matrix (mxm)
      U: Upper triangular matrix (mxn, n=m)
      P: Permutation matrix (mxm)
  """
  # Matrix dim
  m = A.shape[0]
  n = A.shape[1]
  k = np.linalg.matrix_rank(A)
  if m == n and m == k:
    L, D, U, P = q1.DecomposeLDU(A, False)
  # Otherwise, using predefined function 
  else:
    # Here, L^mxk, and U^kxn
    P, L, U = linalg.lu(A)
    # Decompose U into D and U
    D = np.diag(np.diag(U))
    if not np.isclose(np.linalg.det(D), 0.0): 
      U = U / np.diag(D)[:, None]
    else:
      for i in range(k):
        if not np.isclose(D[i, i], 0.0):
          U[i, :] = U[i, :] / D[i, i]

  return L, D, U, P

#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  files = glob.glob(sys.argv[1] + "/*.txt")
  
  for f in files:
    # Read matrix
    A = utils.ReadMatrix(f)
    
    # Matrix dim
    n = A.shape[0] # rows
    m = A.shape[1] # cols
    k = np.linalg.matrix_rank(A)
    print(f"---------------\nA^{m}x{n} with rank: {k}:\n{A}:")

    # LDU decomposition 
    # Using code from q1.py if matrix is square and non-singular
    L, D, U, P = ComputeLDU(A)     
    print(f"\n* LDU Decomposition\nL:\n{L}\nD:\n{D}\nU:\n{U}\nP:\n{P}")

    # SVD decomposition
    U, S, V_T = ComputeSVD(A) 
    print(f"\n* SVD Decomposition\nU:\n{U}\nS:\n{S}\nV_T:\n{V_T}")