# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 14, 2020
#
# @brief      Find rigid body transformation using SVD

#
# Includes ---------------------------------------------------------------------
import sys
import glob
import utils
import q2
import q3
import numpy as np
import scipy
import fractions

# np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})
np.set_printoptions(precision=3, suppress=True)

def P2Matrix(points):
  dim = points.shape[1]
  n_points = points.shape[0]
  cols = (dim + 1) * dim
  rows = n_points * dim
  A = np.zeros((rows, cols))
  for i, p in enumerate(points):
    row = i * dim
    for j in range(dim):
      A[row, j*(1+dim):j*(1+dim)+3] = np.array([p[0], p[1], 1])
      row += 1
  return A

def Q2Vector(Q):
  b = np.zeros((Q.shape[0]*Q.shape[1], 1), dtype=np.float32)
  dim = Q.shape[1]
  row = 0
  for q in Q:
    q = q.reshape((q.shape[0], 1))
    b[row:row+dim] = q
    row += dim
  return b
  
#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  pfile = sys.argv[1] + "/p.txt"
  qfile = sys.argv[1] + "/q.txt"

  p = utils.ReadMatrix(pfile)
  q = utils.ReadMatrix(qfile)
  print(f"p: {p}\nq: {q}")
  assert p.shape == q.shape, f"p.shape({p.shape}) != q.shape({q.shape})"

  A = P2Matrix(p)
  print(f"A\n{A}")
  b = Q2Vector(q)
  print(f"b\n{b}")

  # This is solved through the least-norm approach seen in class
  x, ns, _ = q3.Solve(A, b)
  print(f"x\n{x}")
      