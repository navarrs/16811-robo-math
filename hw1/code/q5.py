# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 14, 2020
#
# @brief      Find rigid body transformation between a set of points P and Q
#             using SVD
# @note:      Used the book Robotics, Vision and Control (Ch.14, pp.505-506) as 
#             refernece

#
# Includes ---------------------------------------------------------------------
import numpy as np
import q2
import scipy
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=4, suppress=True)

#
# Helper methods ---------------------------------------------------------------
def Euler2RotationMatrix(yaw, pitch, roll):
  """
    Converts euler angles to a rotation matrix. 
      *Assumption, the angles are in degrees. 
    Inputs
    ------
      yaw: Rotation about the z-axis.
      pitch: Rotation about the y-axis.
      roll: Rotation about the x-axis.
    Outputs
    -------
      R: 3x3 Rotation matrix. 
  """
  rot = R.from_euler('zyx', [yaw, pitch, roll], degrees=True)
  return rot.as_matrix()

def GenerateRandom3DPoints(n_points, low=-10, high=10):
  """
    Generates a set of Random 3D points in the range [low, high] 
    Inputs
    ------
      n_points: Number of 3D points to generate.
      low: Min value that a point can have.
      high: Max value that a point can have.
      roll: Rotation about the x-axis
    Outputs
    -------
      P: 3xn 3D points.
  """
  P = high * np.random.rand(3, n_points) + low
  return np.array(P)

#
# Problem implementation -------------------------------------------------------
def ComputeRigidTransformation(P, Q):
  """
    Computes the rigid transformation between the corresponding sets P and Q. 
    Inputs
    ------
      P: Set of points p1...pn 
      Q: Set of points q1...qn
    Outputs
    -------
      R: Rotation matrix 3x3
      t: Translation vector 3x1
  """
  # Find centroids 
  pc = np.mean(P, axis=1, dtype=float).reshape(3, 1)
  qc = np.mean(Q, axis=1, dtype=float).reshape(3, 1)

  # Compute moment matrix that encodes the rotation 
  P = P - np.tile(pc, P.shape[1])
  Q = Q - np.tile(qc, P.shape[1])
  W = np.dot(P, Q.T)
  U, _, V_T = q2.ComputeSVD(W)

  # Compute rotation
  rot = np.matmul(V_T.T, U.T)

  # Reflection case
  if np.linalg.det(rot) < 0:
    V_T[2, :] *= -1
    rot = np.matmul(V_T.T, U.T)

  # Compute translation
  t = qc - np.dot(rot, pc)
  return rot, t

#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  import argparse
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--n_points", 
    help="Points to generate", type=int, default=3)
  parser.add_argument("--yaw", 
    help="Rotation on z in degrees", type=int, default=0)
  parser.add_argument("--pitch", 
    help="Rotation on y in degrees", type=int, default=0)
  parser.add_argument("--roll", 
    help="Rotation on x in degrees", type=int, default=0)
  parser.add_argument("--dz", 
    help="Translation on z", type=int, default=0)
  parser.add_argument("--dy", 
    help="Translation on y", type=int, default=0)
  parser.add_argument("--dx", 
    help="Translation on x", type=int, default=0)
  parser.add_argument("--visualize", 
    help="Reading the input from a file", action="store_true", default=False)
  args = parser.parse_args() 

  # Compute some random points (3xN)
  P = GenerateRandom3DPoints(args.n_points)
  # Get user-defined rotation matrix (3x3) and translation vector (3x1)
  rot = Euler2RotationMatrix(args.yaw, args.pitch, args.roll)
  t = np.array([args.dx, args.dy, args.dz]).reshape(3, 1)
 
  # Compute Q = R * P (3xN)
  # @note to self: np.tile is used to sum columns
  Q = np.dot(rot, P) + np.tile(t, args.n_points)
  print(f"P:\n{P}\nQ:\n{Q}")

  rot_est, t_est = ComputeRigidTransformation(P, Q)
  
  print(f"Rot_est\n{rot_est}\nRot_truth\n{rot}")
  print(f"t_est\n{t_est}\nt_truth\n{t}")

  assert np.allclose(rot, rot_est), "R_truth != R_est"
  assert np.allclose(t, t_est), "t_truth != t_est"

  # Plot stuff 
  if args.visualize:
    import matplotlib.pyplot as plt

    Q_est = np.dot(rot_est, P) + np.tile(t, args.n_points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(P[0, :], P[1, :], P[2, :], 
      color='gray', marker='o', label='P')
    ax.scatter(Q[0, :], Q[1, :], Q[2, :], 
      color='green', marker='+', label='Q')
    ax.scatter(Q_est[0, :], Q_est[1, :], Q_est[2, :], 
      color='red', marker='s', label='Qest')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.show()