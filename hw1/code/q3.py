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
from utils import ReadMatrix
import q2
import numpy as np
import scipy

np.set_printoptions(precision=3, suppress=True)

#
# Helpers  ---------------------------------------------------------------------
class SolutionType(Enum):
  UNIQUE_SOL = 1
  ZERO_SOL = 2
  MANY_SOL = 3

def GetAb(filename):
  A = ReadMatrix(f)
  b, A = A[-1, :], A[:-1, :]
  return A, b

#
# Problem implementation -------------------------------------------------------
def Solve(A, b):
  """
    Computes the least squares approach to find x based on SVD
    Input
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
  print(f"\nA^{n}x{m}:\n{A}\nb:\n{b}\n")

  U, S, V_T = q2.ComputeSVD(A) 
  k = np.linalg.matrix_rank(S)
  print(f"\n* SVD Decomposition\nU:\n{U}\nS:\n{S}\nV_T:\n{V_T}")

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
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  files = glob.glob(sys.argv[1] + "/*.txt")
  for f in files:
    print("-----------------------------------")
    # Read matrix
    A, b = GetAb(f)
    # Solve 
    x, ns, sol_type = Solve(A, b)
    b_ = A.dot(x) # Should be the same as b

    # Verify the solutions 
    print(f"\n* Result with {sol_type}:\nx: {x}\nns:{ns}\n")
    if sol_type == SolutionType.UNIQUE_SOL:
      Ax = A.dot(x)
      print(f"Ax == b ? {np.allclose(Ax, b)}\nAx:{Ax} b:{b}")
    elif sol_type == SolutionType.MANY_SOL:
      for xn in ns:
        # generate a real number to make a linear combination with xn
        a = np.random.randint(100)
        # Test for A(x+ a*xn) = b
        Ax = A.dot(x + a * xn)
        print(f"A(x + xn) == b ? {np.allclose(Ax, b)} \nx:{x} a*xn:{a * xn} Ax:{Ax} b{b}")