# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 4, 2020
#
# @brief      Implementation of the LDU decomposition to solve the equation  
#                                     Ax = b 

#
# Includes --------------------------------------------------------------------
import numpy as np
from numpy.linalg import matrix_rank, inv

#
# Function implementation ------------------------------------------------------
def DecomposeLA(A):
  assert A.shape[0] == A.shape[1] and matrix_rank(A) == A.shape[0], \
    "Input matrix is singular"

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
    
    



#
# Main program -----------------------------------------------------------------
def main():
  #A = np.random.rand(3, 3)
  A = np.matrix([[1, -2, 1], [1, 2, 2], [2, 3, 4]], dtype=float)
  L, D, U = DecomposeLA(A)
  print("L: ", L, "\nD: ", D, "\nU: ", U)

if __name__ == "__main__":
  main()