# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 4, 2020
#
# @brief      Computing LDU and SVD decomposition 

#
# Includes ---------------------------------------------------------------------
import argparse
import sys 
sys.path.insert(1, '../')

from hw1.utils import utils
from hw1.q1 import q1
from numpy.linalg import svd, lu

#
# Implementation ---------------------------------------------------------------
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--file", help="File containing the matrix", type=str, default="a1.txt")
  parser.add_argument("--type", help="Matrix type (int float)", default='f')
  args = parser.parse_args()

  A = utils.ReadMatrix(args.file, args.type)

  print("Matrix A:\n {}".format(A))
  if A.shape[0] == A.shape[1]:
    L, D, U, P = q1.DecomposeLDU(A)
    print("Matrix Decomposition\nL:\n{}\nD:\n{}\nU:\n{}\np:\n{}"
      .format(L, D, U, P))
  else:
    P, L, DU = lu(A)
    D = DU.diagonal() * np.identity(A.shape[0])
    U = inv(D).dot(DU)
    
    print("Matrix Decomposition\nL:\n{}\nD:\n{}\nU:\n{}\np:\n{}"
      .format(L, D, U, P))