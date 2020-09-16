# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 4, 2020
#
# @brief      Implementation of the LDU decomposition to solve the equation 
#                                   Ax = b 
# @notes      This code is designed to work with square, non-singular matrices. 

#
# Includes ---------------------------------------------------------------------
import argparse
import numpy as np
from numpy.linalg import matrix_rank, inv
from utils import ReadMatrix

#
# Problem implementation -------------------------------------------------------
def DecomposeLDU(A, do_assert=False):
  """
    Performs LDU decomposition via Gaussian reduction. This code is designed for 
    square, non-singular matrices. 
    Inputs
    ------
      A: Square (mxn, n=m), non-singular matrix that will be decomposed
      do_assert: if true, it check if PA == LDU
    Outputs
    -------
      L: Low triangular matrix (mxm)
      D: Diagonal matrix (mxm)
      U: Upper triangular matrix (mxn, n=m)
      P: Permutation matrix (mxm)
  """
  assert A.shape[0] == A.shape[1] and A.shape[0] == np.linalg.matrix_rank(A), \
    "Invalid: Non-square or singular matrix"

  if do_assert:
    # Create a copy of A to test the resulting matrices
    A_ = A.copy()

  n = A.shape[0]
  P = np.identity(n)
  L = np.identity(n)
  D = np.identity(n)
  U = np.identity(n)

  # 1. Perform Gauss Reduction 
  for i in range(n):

    # Identify if a permutation is required based on the index of the max element 
    # in the ith column
    si = [abs(A[j, i]) for j in range(i, n)]
    max_i = si.index(max(si)) + i
  
    # Permute if max_i is not equal to current row
    if max_i != i:
      A[[max_i, i]] = A[[i, max_i]]
      P[[max_i, i]] = P[[i, max_i]]
    
    # Gaussian reduction on A
    for j in range(i+1, n):
      A[j, i] /= A[i, i] # a
      # Update jth A row as: 
      #    rowj = rowj - a * rowi
      for k in range(i+1, n):
        A[j, k] -=  A[j, i] * A[i, k]

  # 2. Decompose L, D, U
  for i in range(n):
    for j in range(i):
      L[i, j], A[i, j] = A[i, j], 0.0
  D = np.diag(np.diag(A))
  U = A / np.diag(D)[:, None]

  # 3. If specified verify the result
  if do_assert:
    assert np.allclose(P.dot(A_), (L.dot(D)).dot(U)), "PA != LDU"
    print("Success: PA == LDU")
    print(f"\nL:\n{L}\nD:\n{D}\nU:\n{U}\nP:\n{P}\n")

  return L, D, U, P

def Solve(A, b, do_assert):
  """
    Solves an equation of the form Ax = b via LDU decomposition. This is done in 
    three main steps:
      (1) Get L, D, U, P matrices
      (2) Solve for y: Ly = b'
      (3) Sove for x: Ux = D^-1 * y
    Inputs
    ------
      A: Square (nxn), non-singular matrix that will be decomposed
      b: Vector (nx1)
      do_assert: If enabled, verifies that Ax == b
    Outputs
    -------
      x: Vector (nx1) that is a solution to Ax = b
  """
  if do_assert:
    A_ = A.copy()
    b_ = b.copy()

  # Decompose matrix A
  n = A.shape[0]
  L, D, U, P = DecomposeLDU(A, do_assert)
  
  x = np.zeros(n)
  y = np.zeros(n)

  # Solve Ly = b' using forward substitution. Example:
  # ( 1  0  0 ) (y1)   (b1')        y1 = b1'
  # ( a  1  0 ) (y2) = (b2')  --->  y2 = b2' - a * y1
  # ( b  c  1 ) (y3)   (b3')        y3 = b3' - b * y1 - c * y3
  b = P.dot(b)
  for i in range(n):
    y[i] = b[i]
    for j in range(i):
      y[i] -= L[i, j] * y[j]
  
  # Solve Ux = D^-1 * y using back substitution. Example:
  # ( 1  a  b ) (x1)   (y1')        x1 = y1' - a * x1 - b * x3
  # ( 0  1  c ) (x2) = (y2')  --->  x2 = y2' - c * x1
  # ( 0  0  1 ) (x3)   (y3')        x3 = y3' 
  y = inv(D).dot(y)
  for i in reversed(range(n)):
    x[i] = y[i]
    for j in range(n-1, i, -1):
      x[i] -= U[i, j] * x[j]
  
  if do_assert:
    # Verify that Ax leads to b
    Ax = A_.dot(x)
    assert np.allclose(Ax, np.transpose(b_)), "Ax != b"
    print("Success: Ax == b")
    print(f"\nAx:\n{Ax}\nb:\n{np.transpose(b_)}\n")
  return x

#
# Helper methods ---------------------------------------------------------------
def SolveRandom(do_assert=False, min_n=2, max_n=10, min_val=-100, max_val=100):
  """
    Creates a random matrix A(nxn) and vector b(nx1). Then, solves Ax = b for x
    using LDU. 
    Inputs
    ------
      do_assert: If enabled, verifies that Ax == b
      min_n: Min size of square matrix A and vector b
      max_n: Max size of square matrix A and vector b
      min_val: Min allowed value to create the matrix
      max_val: Max allowed value to create the matrix
    Outputs
    -------
      Nothing
  """
  n = np.random.randint(low=min_n, high=max_n)
  A = np.matrix(np.random.uniform(low=min_val, high=max_val, size=(n, n)))
  b = np.matrix(np.random.uniform(low=min_val, high=max_val, size=(n, 1)))
  print(f"A:\n{A}\nb:\n{b}\n")
  x = Solve(A, b, do_assert)
  print(f"x:{x}\n")

def SolveExample(A, b, do_assert):
  """
    Loads a system of linear equations from a path and solves Ax = b for x using 
    LDU decomposition.  
    Inputs
    ------
      do_assert: If enabled, verifies that Ax == b
    Outputs
    -------
      Nothing
  """
  # Assume last row in matrix is vector b
  # A = ReadMatrix(path)
  # b, A = A[-1, :], A[:-1, :]
  print(f"A:\n{A}\nb:\n{b}\n")

  assert A.shape[0] == A.shape[1] and A.shape[0] == b.shape[0], \
    "Dimensions of A and b are nxn and nx1, but are A:{}x{} b:{}x1".format(
      A.shape[0],A.shape[1],b.shape[0])
  
  x = Solve(A, b, do_assert)
  print(f"x:{x}\n")

def GetExamples():
  A1 = np.array([[1, -2, 1], [1, 2, 2], [2, 3, 4]], dtype=float)
  b1 = np.array([5, 9, 2],  dtype=float)

  A2 = np.array([[2, 3, 3], [1, -2, 1], [3, -1, -2]],  dtype=float)
  b2 = np.array([5, -4, 3], dtype=float)

  A3 = np.array([[1,2,1,4], [2,0,4,3], [4,2,2,1], [-3,1,3,2]],  dtype=float)
  b3 = np.array([13,28,20,6], dtype=float)

  A4 = np.array([[2,-1,4,1,-1], [-1,3,-2,-1,2], [5,1,3,-4,1], 
                 [3,-2,-2,-2,3], [-4,-1,-5,3,-4]],  dtype=float)
  b4 = np.array([7,1,33,24,-49], dtype=float)

  A = [A1, A2, A3, A4]
  b = [b1, b2, b3, b4]
  return A, b

#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # Use it to verify PA == LDU and Ax == b
  parser.add_argument("--do_assert", help="Asserts that PA == LDU and Ax == b", 
    action="store_true", default=False)
  # If testing a sample from file, add --from-file --path /path/to/file
  parser.add_argument("--random", help="Reading the input from a file", 
    action="store_true", default=False)
  args = parser.parse_args()

  if args.random:
    SolveRandom(args.do_assert)
  else:
    A, b = GetExamples()
    for i in range(len(A)):
      A_ = A[i]
      b_ = b[i]
      SolveExample(A_, b_, args.do_assert)