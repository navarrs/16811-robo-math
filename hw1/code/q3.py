# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 10, 2020
#
# @brief      Solving systems of linear equations Ax = b via SVD. 

#
# Includes ---------------------------------------------------------------------
import sys
import glob
from enum import Enum
import utils
import q2
import numpy as np
import scipy

np.set_printoptions(precision=3, suppress=True)

class SolutionType(Enum):
  UNIQUE_SOL = 1
  ZERO_SOL = 2
  MANY_SOL = 3

#
# Functions --------------------------------------------------------------------
def GetAb(filename):
  A = utils.ReadMatrix(f)
  b, A = A[-1, :], A[:-1, :]
  return A, b

def GetNullSpace(A):
  """
    Calculates the nullspace
  """

def Solve(A, b):
  """
    Solves the Ax = b via SVD
    Input
    -----
      A: mxn matrix
      b: 1xn vector
    Output
    ------
      x: The solution to the system of linear equations. 
      ns: Null space of A
      sol_type: If the system has UNIQUE, ZERO or MANY solutions
  """
  n = A.shape[0] # rows
  m = A.shape[1] # cols
  print(f"\n---------------\nA^{n}x{m}:\n{A}\nb:\n{b}\n")

  U, S, V_T = q2.ComputeSVD(A) 
  k = np.linalg.matrix_rank(S)
  print(f"\n* SVD Decomposition\nU:\n{U}\nS:\n{S}\nV_T:\n{V_T}")

  # This is solved through the least-norm approach seen in class
  U_T = np.transpose(U)
  S_inv = S
  for i in range(k):
    S_inv[i, i] = 1 / S_inv[i, i]
  V = np.transpose(V_T)

  # x = V 1/S U_T b
  x  = np.matmul(V, np.matmul(S_inv, np.dot(U_T, b)))

  # If b in colspace(A), then Ax = b with x = V 1/S U_T b
  # Check if the type of solution to the system
  b_ = np.matmul(A, x)
  if k < n:
    ns = V[:, :n-k]
    if np.allclose(b, b_):
      # b is in colspace(A) but there are many solutions
      return x, ns, SolutionType.MANY_SOL
    else:
      # b is not in colspace(A)
      return np.nan, ns, SolutionType.ZERO_SOL

  print(f"\nb_{b_}\n")
  return x, V[:, :n-k], SolutionType.UNIQUE_SOL

#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  files = glob.glob(sys.argv[1] + "/*.txt")
  for f in files:
    # Read matrix
    A, b = GetAb(f)
    # Solve 
    x, ns, sol_type = Solve(A, b)
    b_ = A.dot(x) # Should be the same as b
    print(f"\n* Result with {sol_type}:\nx: {x}\nnull space: {ns}\n")