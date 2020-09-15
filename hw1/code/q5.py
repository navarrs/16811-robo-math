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
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
import argparse
import matplotlib.pyplot as plt

# np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})
np.set_printoptions(precision=3, suppress=True)

def ComputeRotationMatrix(yaw, pitch, roll):
  rot = R.from_euler('zyx', [yaw, pitch, roll], degrees=True)
  return rot.as_matrix()

def GenerateRandom3DPoints(n_points, low=0, high=10, type=int):
  return np.random.randint(low=low, high=high, size=(n_points, 3))

def Rotate(rot, p):
  return np.dot(rot, p)
  
#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--n_points", help="Points to generate", 
    type=int, default=3)
  parser.add_argument("--yaw", help="Rotation on z in degrees", 
    type=int, default=0)
  parser.add_argument("--pitch", help="Rotation on y in degrees", 
    type=int, default=0)
  parser.add_argument("--roll", help="Rotation on x in degrees", 
    type=int, default=0)
  args = parser.parse_args() 

  # Compute some random points
  # N x 3
  # P = GenerateRandom3DPoints(args.n_points)
  # print(P, P[:, 0])
  P = np.array([[0,0,0], [1,0,1], [0,1,0]])
  p_center = P[0]
  print(f"P\n{P}\np_center: {p_center}\n")
  # Shift points 
  Ps = P - p_center

  # Get rotation matrix
  # 3 x 3
  rot = ComputeRotationMatrix(args.yaw, args.pitch, args.roll)
  print(f"Rotation\n{rot}\n")

  # Compute Q = R P^T
  # 3 x N
  Q = np.matmul(rot, np.transpose(P))
  Q = np.transpose(Q)
  Qs = Q + p_center
  print(f"Q\n{Q}\n")
  #Q = np.matmul(P, rot)

  # # Plot stuff 
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  ax.plot3D(P[:, 0], P[:, 1], P[:, 2], color='gray', marker='s', label='P')
  ax.plot3D(Ps[:, 0], Ps[:, 1], Ps[:, 2], color='blue', marker='o', label='Ps')
  ax.plot3D(Q[:, 0], Q[:, 1], Q[:, 2], color='red', marker='o', label='Q')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.legend()
  plt.show()


      