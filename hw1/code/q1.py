# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 4, 2020
#
# @brief      Implementation of the LDU decomposition to solve the equation  
#                                   Ax = b 

#
# Includes ---------------------------------------------------------------------
import argparse
import numpy as np
from numpy.linalg import matrix_rank, inv
import utils

#
# LDU Decomposition implementation ---------------------------------------------
def DecomposeLDU(A, do_assert=False):
  """
    Performs LDU decomposition via Gaussian reduction.
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
    "Input matrix is invalid"

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
      # pivot
      A[j, i] /= A[i, i]
      # Update jth A row as: 
      #    rowj = rowj - pivot * rowi
      for k in range(i+1, n):
        A[j, k] -=  A[j, i] * A[i, k]

  # 2. Decompose L, D, U
  for i in range(n):
    for j in range(i):
      L[i, j], A[i, j] = A[i, j], 0.0
  D = np.diag(np.diag(A))
  U = A / np.diag(D)[:, None]

  # 3. If specified check result
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
      (3) Sove for x: Ux = Dinv * y
    Inputs
    ------
      A: Square (nxn), non-singular matrix that will be decomposed
      b: Vector (nx1)
    Outputs
    -------
      x: Vector (nx1) that is a solution to Ax = b
  """
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
  return x

#
# Main program -----------------------------------------------------------------
def solve_random(size, min_val, max_val, do_assert):
  A = np.matrix(np.random.uniform(low=min_val, high=max_val, size=(size, size)))
  b = np.matrix(np.random.uniform(low=min_val, high=max_val, size=(size, 1)))
  print(f"A:\n{A}\nb:\n{b}\n")

  if do_assert:
    A_ = A.copy()
    b_ = b.copy()

  x = Solve(A, b, do_assert)
  if do_assert:
    # Verify that Ax leads to b
    assert np.allclose(A_.dot(x), np.transpose(b_)), "Ax != b"
  print(f"x:{x}\n")

def solve_from_file(path, do_assert):
  # Assume last row in matrix is vector b
  A = utils.ReadMatrix(path)
  b, A = A[-1, :], A[:-1, :]
  print(f"A:\n{A}\nb:\n{b}\n")

  assert A.shape[0] == A.shape[1] and A.shape[0] == b.shape[0], \
    "Dimensions of A and b are nxn and nx1, but are A:{}x{} b:{}x1".format(
      A.shape[0],A.shape[1],b.shape[0])
  
  if do_assert:
    A_ = A.copy()

  x = Solve(A, b, do_assert)
  if do_assert:
    # Verify that Ax leads to b
    assert np.allclose(A_.dot(x), b), "Ax != b"

  print(f"x:{x}\n")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--do_assert", 
    help="Asserts that PA == LDU", action="store_true", default=False)

  # If --from_file and --path are specified, the other configuration parameters 
  # are ignored
  parser.add_argument("--from_file", 
    help="Reading the input from a file", action="store_true", default=False)
  parser.add_argument("--path", 
    help="File containing matrix A and vector b", default="Ab.txt")

  # These parameters are used to create a random system of equations. 
  parser.add_argument("--size", 
    help="matrix size", type=int, default=3)
  parser.add_argument("--min", 
    help="min value in matrix", type=float, default=-100.0)
  parser.add_argument("--max", 
    help="max value in matrix", type=float, default=100.0)
  args = parser.parse_args()

  if args.from_file:
    solve_from_file(args.path, args.do_assert)
  else:
    solve_random(args.size, args.min, args.max, args.do_assert)