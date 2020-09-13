# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 4, 2020
#
# @brief      Unit Test for q1.py

import argparse
import numpy as np
import q1

class TestLDU():
  def __init__(self, n_min=2, n_max=100, low=-10000.0, high=10000.0):
    # Max matrix size
    self.n_min = n_min
    # Max matrix size
    self.n_max = n_max
    # Min matrix value
    self.low = low
    # Max matrix value
    self.high = high

  def TestDecompositionLDU(self):
    n = np.random.randint(low=self.n_min, high=self.n_max)
    print("Testing q1.DecompositionLDU with Matrix of size {}x{}".format(n, n))
    A = np.matrix(np.random.uniform(low=self.low, high=self.high, size=(n, n)))
    A_ = A.copy()
    L, D, U, P = q1.DecomposeLDU(A)
    PA = P.dot(A_)
    LDU = (L.dot(D)).dot(U)
    assert np.allclose(PA, LDU), "PA != LDU"
  
  def TestSolve(self):
    n = np.random.randint(low=self.n_min, high=self.n_max)
    print("Testing q1.Solve with n = {}".format(n))
    A = np.matrix(np.random.uniform(low=self.low, high=self.high, size=(n, n)))
    A_ = A.copy()
    b = np.matrix(np.random.uniform(low=self.low, high=self.high, size=(n, 1)))
    b_ = b.copy()
    x = q1.Solve(A, b, True)
    assert np.allclose(A_.dot(x), np.transpose(b_)), "Ax != b"
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--n_tests", help="tests to execute", type=int, default=10)
  args = parser.parse_args()

  test = TestLDU()

  for t in range(args.n_tests):
    print("Test {}".format(t))
    test.TestDecompositionLDU()
    test.TestSolve()
  print("All tests passed")