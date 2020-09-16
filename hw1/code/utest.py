# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 4, 2020
#
# @brief      Unit Test for q1.py

import argparse
import numpy as np
import q1
import q5

def TestSpecific():
  A = np.matrix([[0, 3], [2, 0]])
  A_ = A.copy()
  L, D, U, P = q1.DecomposeLDU(A)
  print(f"P\n{P}\nA\n{A_}\nL\n{L}\n{D}\nU{U}\n")
  PA = P.dot(A_)
  LDU = (L.dot(D)).dot(U)
  print(f"PA\n{PA}\nLDU\n{LDU}\n")
  assert np.allclose(PA, LDU), "PA != LDU"

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
  
class TestRigidMotion():
  def __init__(self, N_min=3, N_max=1000, deg_min=-180, deg_max= 180, d_min=-100, d_max=100):
    # Point set size
    self.N_min = N_min
    self.N_max = N_max
    # Angle values
    self.deg_min = deg_min
    self.deg_max = deg_max
    # Displacement values
    self.d_min = d_min
    self.d_max = d_max
  
  def TestRM(self):
    n = np.random.randint(low=self.N_min, high=self.N_max)
    yaw = np.random.randint(low=self.deg_min, high=self.deg_max)
    pitch = np.random.randint(low=self.deg_min, high=self.deg_max)
    roll = np.random.randint(low=self.deg_min, high=self.deg_max)
    dx = np.random.randint(low=self.d_min, high=self.d_max)
    dy = np.random.randint(low=self.d_min, high=self.d_max)
    dz = np.random.randint(low=self.d_min, high=self.d_max)
    
    # Test
    R_truth = q5.Euler2RotationMatrix(yaw, pitch, roll)
    t_truth = np.array([dx, dy, dz]).reshape(3, 1)
    P, Q_truth = q5.ComputePointSet(R_truth, t_truth, n)
    R_est, t_est, _ = q5.ComputeRigidTransformation(P, Q_truth)

    assert np.allclose(R_truth, R_est), "R_truth != R_est"
    assert np.allclose(t_truth, t_est), "t_truth != t_est"
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--n_tests", help="tests to execute", type=int, default=10)
  parser.add_argument("--ldu", help="Tests LDU", action="store_true", default=False)
  parser.add_argument("--rigid", help="Tests Rigid Motion", action="store_true", default=False)
  args = parser.parse_args()

  # Test LDU
  if args.ldu:
    print("Testing LDU")
    test = TestLDU()
    for t in range(args.n_tests):
      print("Test {}".format(t))
      test.TestDecompositionLDU()
      print("\t--LDU success")
      test.TestSolve()
      print("\t--Solve success")
    TestSpecific()
  
  if args.rigid:
    print("Testing Rigid Motion")
    test = TestRigidMotion()
    for t in range(args.n_tests):
      print("Test {}".format(t))
      test.TestRM()
      print("\t--RM success")

  print("All tests passed")