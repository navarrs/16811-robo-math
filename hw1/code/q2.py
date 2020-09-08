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

#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  files = glob.glob(sys.argv[1] + "/*.txt")
  
  for i, f in enumerate(files):
    # Read matrix
    A = utils.ReadMatrix(f, float)
    print(f"---------------\nMatrix A:\n{A}:")

    # LDU decomposition 
    if A.shape[0] == A.shape[1]:
      L, D, U, P = q1.DecomposeLDU(A)
      print(f"LDU Decomposition\nL:\n{L}\nD:\n{D}\nU:\n{U}\nP:\n{P}")
    else:
      P, L, DU = linalg.lu(A)
      print(f"LU Decomposition\nL:\n{L}\nU:\n{DU}\nP:\n{P}")
    
    # SVD decomposition
    U, S, V = linalg.svd(A)
    print(f"SVD Decomposition\nU:\n{U}\nS:\n{S}\nV:\n{V}")