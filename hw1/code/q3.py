# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 10, 2020
#
# @brief      Solving systems of linear equations Ax = b via SVD. 

#
# Includes ---------------------------------------------------------------------
import sys
import glob

import utils
import q2
import numpy as np

np.set_printoptions(precision=3, suppress=True)

#
# Functions --------------------------------------------------------------------
def GetAb(filename):
  A = utils.ReadMatrix(f)
  b, A = A[-1, :], A[:-1, :]
  return A, b

def Solve(A, b):
  """
    Solves the Ax = b via SVD
    Input
    -----
      A: mxn matrix
      b: 1xn vector
    Output
    ------
      x: the solution to the system of linear equations. 
  """
  m = A.shape[0]
  n = A.shape[1]
  k = np.linalg.matrix_rank(A)
  print(f"\n---------------\nA^{m}x{n} with rank: {k}:\n{A}\nb:\n{b}\n")

  U, S, V_T = q2.ComputeSVD(A) 
  print(f"\n* SVD Decomposition\nU:\n{U}\nS:\n{S}\nV_T:\n{V_T}")

  # Least squares solution 
  U_T = np.transpose(U)
  S_inv = S
  for i in range(k):
    S_inv[i, i] = 1 / S_inv[i, i]
  V = np.transpose(V_T)
  print(f"\n* V 1/S U_T Decomposition\nV:\n{V}\n1/S:\n{S_inv}\nU_T:\n{U_T}")

  # x = V 1/S U_T b
  x = np.matmul(V, np.matmul(S_inv, np.dot(U_T, b)))
  b_ = A.dot(x) # Should be the same as b
  print(f"\n* Result x:\n{x}\nb_:\n{b_}\n")
  return 0

#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  files = glob.glob(sys.argv[1] + "/*.txt")
  for f in files:
    # Read matrix
    A, b = GetAb(f)
    # Solve 
    x = Solve(A, b)
