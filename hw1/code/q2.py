# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 4, 2020
#
# @brief      Computing LDU and SVD decomposition 
#                  A  =  L   D  U
#                 mxn = mxm mxm mxn
#                  A  =  U   S  V_t
#                 mxn = mxm mxn nxn

#
# Includes ---------------------------------------------------------------------
import sys
import glob

import utils
import q1
import numpy as np
from scipy import linalg 

#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  files = glob.glob(sys.argv[1] + "/*.txt")
  
  for f in files:
    # Read matrix
    A = utils.ReadMatrix(f)
    
    # Matrix dim
    m = A.shape[0]
    n = A.shape[1]
    k = np.linalg.matrix_rank(A)
    print(f"---------------\nA^{m}x{n} with rank: {k}:\n{A}:")

    # LDU decomposition 
    # Using code from q1.py if matrix is square and non-singular
    if m == n and m == k:
      L, D, U, P = q1.DecomposeLDU(A, False)
      print(f"\n* LDU Decomposition\nL:\n{L}\nD:\n{D}\nU:\n{U}\nP:\n{P}")
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
            
      print(f"\n* (Predefined) LDU Decomposition\nL:\n{L}\nD:\n{D}\nU:\n{U}\nP:\n{P}")

    # SVD decomposition
    U, s, V = linalg.svd(A)
    S = linalg.diagsvd(s, m, n)
    print(f"\n* SVD Decomposition\nU:\n{U}\nS:\n{S}\nV:\n{V}")