# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 10, 2020
#
# @brief      Solving systems of linear equations Ax = b via SVD. 

#
# Includes ---------------------------------------------------------------------
from enum import Enum
import q2
import numpy as np
import scipy
#from utils import ReadMatrix


#
# Helpers ----------------------------------------------------------------------
np.set_printoptions(precision=3, suppress=True)

class SolutionType(Enum):
  UNIQUE_SOL = 1
  ZERO_SOL = 2
  MANY_SOL = 3

def GetExamples():
  """
    Test examples
    Inputs
    ------
      None
    Outputs
    -------
      A, b: List of square matrices A and corresponding vectors b
  """
  A1 = np.array([[1, 1, 1], [10, 2, 9], [8, 0, 7]], dtype=float)
  b1 = np.array([1, 3, 1],  dtype=float)

  A2 = np.array([[1, 1, 1], [10, 2, 9], [8, 0, 7]], dtype=float)
  b2 = np.array([3, 2, 2],  dtype=float)

  A3 = np.array([[10, -10, 0], [0, -4, 2], [2, 0, -5]], dtype=float)
  b3 = np.array([10, 2, 13],  dtype=float)

  A = [A1, A2, A3]
  b = [b1, b2, b3]
  return A, b

#
# Problem implementation -------------------------------------------------------
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
  A_list, b_list = GetExamples()
  for i in range(len(A_list)):
    print("\n\n-----------------------------------")
    # Read matrix
    A, b = A_list[i], b_list[i]
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