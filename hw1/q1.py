# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 4, 2020
#
# @brief      Implementation of the LDU decomposition to solve the equation  
#                                     Ax = b 

#
# Includes ---------------------------------------------------------------------
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
  """
  assert A.shape[0] == A.shape[1] and matrix_rank(A) == A.shape[0], \
    "Input matrix is invalid"
  L = np.identity(A.shape[0])
  D = np.identity(A.shape[0])
  U = np.identity(A.shape[0])

  for i in range(A.shape[0]):
    pivot = A[i, i]

    # Get L values below diagonal for column i
    for j in range(i+1, A.shape[1]):
      L[j, i] = A[j, i] / pivot

      # Update jth A row as: 
      #    rowj = rowj - Lji * rowi
      for k in range(i, A.shape[0]):
        A[j, k] -= L[j, i] * A[i, k]
    
    # Update diagonal value dii 
    D[i, i] = A[i, i]
  
  U = inv(D) * A
  return L, D, U

def Solve(A, b):
  """
    Solves an equation of the form Ax = b via LDU decomposition. This is done in 
    three main steps:
      (1) Get L, D, U matrices
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
  L, D, U = DecomposeLDU(A)
  #print("L: ", L, "\nD: ", D, "\nU: ", U)
  
  x = np.zeros(A.shape[0])
  y = np.zeros(A.shape[0])

  # Solve Ly = b' using forward substitution, e.g:
  # ( 1  0  0 ) (y1)   (b1')        y1 = b1'
  # ( a  1  0 ) (y2) = (b2')  --->  y2 = b2' - a * y1
  # ( b  c  1 ) (y3)   (b3')        y3 = b3' - b * y1 - c * y3
  for i in range(y.shape[0]):
    y[i] = b[i]
    for j in range(i):
      y[i] -= L[i, j] * y[j]
  # print("y: ", y)
  
  # Solve Ux = D^-1 * y using backward substitution
  # ( 1  a  b ) (x1)   (y1')        x1 = y1' - a * x1 - b * x3
  # ( 0  1  c ) (x2) = (y2')  --->  x2 = y2' - c * x1
  # ( 0  0  1 ) (x3)   (y3')        x3 = y3' 
  y = inv(D).dot(y)
  for i in reversed(range(y.shape[0])):
    x[i] = y[i]
    for j in reversed(range(i)):
      x[i] -= U[i, j] * x[i]
  #print("x: ", x)

  return x

#
# Main program -----------------------------------------------------------------
def main():
  #A = np.random.rand(3, 3)
  A = np.matrix([[1, -2, 1], [1, 2, 2], [2, 3, 4]], dtype=float)
  b = np.array([5, 9, 2], dtype=float)
  x = Solve(A, b)

if __name__ == "__main__":
  main()