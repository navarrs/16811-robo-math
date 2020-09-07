# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 4, 2020
#
# @brief      Implementation of the LDU decomposition to solve the equation  #   
#                                   Ax = b 

#
# Includes ---------------------------------------------------------------------
import argparse
import numpy as np
from numpy.linalg import matrix_rank, inv

#
# Function implementation ------------------------------------------------------
def DecomposeLDU(A):
  """
    Performs LDU decomposition via Gaussian reduction.
    Inputs
    ------
      A: Square (mxn, n=m), non-singular matrix that will be decomposed
    Outputs
    -------
      L: Low triangular matrix (mxm)
      D: Diagonal matrix (mxm)
      U: Upper triangular matrix (mxn, n=m)
      P: Permutation matrix (mxm)
  """
  assert A.shape[0] == A.shape[1] and matrix_rank(A) == A.shape[0], \
    "Input matrix is invalid"

  n = A.shape[0]
  P = np.identity(n)
  L = np.identity(n)
  D = np.identity(n)
  U = np.identity(n)

  for i in range(n):
    # Identify if a permutation is required
    # q: scaled values in the ith column 
    q = [0.0 for i in range(i, n)]
    idx = 0
    for j in range(i, n):
      # s: max values per row (from i to n)
      s = 0.0
      for k in range(i, n):
        v = abs(A[j, k])
        s = v if v > s else s
      q[idx] = abs(A[j, i]) / s
      idx += 1
    max_i = q.index(max(q)) + i
  
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

  # Decompose L and D
  for i in range(n):
    for j in range(n):
      if i == j:
        D[i, i] = A[i, j]
      elif i > j:
        L[i, j], A[i, j] = A[i, j], 0.0

  # Decompose DU
  U = inv(D).dot(A)
  return L, D, U, P

def Solve(A, b):
  """
    Solves an equation of the form Ax = b via LDU decomposition. This is done in 
    three main steps:
      (1) Get L, D, U, P matrices
      (2) Solve for y: Ly = b'
      (3) Sove for x: Ux = Dinv * y
    Inputs
    ------
      A: Square (mxn, n=m), non-singular matrix that will be decomposed
      b: Vector (nx1)
    Outputs
    -------
      x: Vector (nx1) that solves Ax = b
  """
  # Decompose matrix A
  n = A.shape[0]
  L, D, U, P = DecomposeLDU(A)
  
  x = np.zeros(n)
  y = np.zeros(n)

  # Solve Ly = b' using forward substitution, e.g:
  # ( 1  0  0 ) (y1)   (b1')        y1 = b1'
  # ( a  1  0 ) (y2) = (b2')  --->  y2 = b2' - a * y1
  # ( b  c  1 ) (y3)   (b3')        y3 = b3' - b * y1 - c * y3
  b = P.dot(b)
  for i in range(n):
    y[i] = b[i]
    for j in range(i):
      y[i] -= L[i, j] * y[j]
  
  # Solve Ux = D^-1 * y using back substitution
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
def main(size, min_val, max_val):
  A = np.matrix(np.random.uniform(low=min_val, high=max_val, size=(size, size)))
  b = np.matrix(np.random.uniform(low=min_val, high=max_val, size=(size, 1)))
  x = Solve(A, b)
  print("x: ", x)
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--size", help="matrix size", type=int, default=5)
  parser.add_argument("--min_val", help="min value in matrix", type=float, default=-100.0)
  parser.add_argument("--max_val", help="max value in matrix", type=float, default=100.0)
  args = parser.parse_args()
  main(args.size, args.min_val, args.max_val)