# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 4, 2020
#
# @brief      Computing LDU and SVD decomposition for a given set of matrices

#
# Includes ---------------------------------------------------------------------
import numpy as np
import q1
from scipy import linalg 
#from utils import ReadMatrix#, LatexArr

np.set_printoptions(precision=3, suppress=True)

#
# Problem implementation -------------------------------------------------------
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

def GetExamples():
  A1 = np.array([[10, -10, 0], [0, -4, 2], [2, 0, -5]], 
    dtype=float)
  A2 = np.array([[5, -5, 0, 0], [5, 5, 5, 0], [0, -1, 4, 1],
                 [0, 4, -1, 2], [0, 0, 2, 1]],  dtype=float)
  A3 = np.array([[1, 1, 1], [10, 2, 9], [8, 0, 7]],  dtype=float)
  A = [A1, A2, A3]
  return A

def ComputeLDU(A):
  """
    Performs LDU decomposition via Gaussian reduction. Uses two methods: 
      (1) DecomposeLDU() - from q1. Used for square, non-singular matrices. 
      (2) linalg.lu - from Scipy. Used for the remaining cases. 
    Input
    ------
      A: Matrix (mxn)
    Output
    ------
      L: Low triangular matrix (mxm)
      D: Diagonal matrix (mxm)
      U: Upper triangular matrix (mxn)
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
  A_ = GetExamples()
  
  for i in range(len(A_)):
    # Read matrix
    # A = ReadMatrix(f)
    A = A_[i]

    # Matrix dim
    n = A.shape[0] # rows
    m = A.shape[1] # cols
    k = np.linalg.matrix_rank(A)
    print(f"\n\n---------------\nA^{m}x{n} with rank: {k}:\n{A}:")

    # SVD decomposition
    print(f"\n* SVD Decomposition")
    U, S, V_T = ComputeSVD(A) 
    print(f"U:\n{U}\nS:\n{S}\nV_T:\n{V_T}")
    #print(f"U:\n{LatexArr(U)}\nS:\n{LatexArr(S)}\nV_T:\n{LatexArr(V_T)}")

    # LDU decomposition 
    # Using code from q1.py if matrix is square and non-singular
    print(f"\n* LDU Decomposition")
    L, D, U, P = ComputeLDU(A)     
    print(f"L:\n{L}\nD:\n{D}\nU:\n{U}\nP:\n{P}")
    #print(f"L:\n{LatexArr(L)}\nD:\n{LatexArr(D)}\nU:\n{LatexArr(U)}\nP:\n{LatexArr(P)}")
