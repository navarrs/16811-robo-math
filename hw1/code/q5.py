# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 14, 2020
#
# @brief      Find rigid body transformation using SVD

#
# Includes ---------------------------------------------------------------------
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
import q2

np.set_printoptions(precision=4, suppress=True)

def ComputeRotationMatrix(yaw, pitch, roll):
  rot = R.from_euler('zyx', [yaw, pitch, roll], degrees=True)
  return rot.as_matrix()

def GenerateRandom3DPoints(n_points, low=-10, high=10, type=int):
  P = high * np.random.rand(3, n_points) + low
  # P = np.random.randint(low=low, high=high, size=(3,n_points))
  return np.array(P)

def Rotate(rot, p):
  return np.dot(rot, p)

def ComputeRigidTransformation(P, Q):
  # Find centroids 
  pc = np.mean(P, axis=1, dtype=float).reshape(3, 1)
  qc = np.mean(Q, axis=1, dtype=float).reshape(3, 1)

  # Compute moment matrix that encodes the rotation 
  P = P - np.tile(pc, P.shape[1])
  Q = Q - np.tile(qc, P.shape[1])
  W = np.dot(P, Q.T)
  U, _, V_T = q2.ComputeSVD(W)
  rot = np.matmul(V_T.T, U.T)

  # Handle special case 
  if np.linalg.det(rot) < 0:
    V_T[2, :] *= -1
    rot = np.matmul(V_T.T, U.T)
  
  t = qc - np.dot(rot, pc)
  return rot, t


#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  import argparse
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--n_points", help="Points to generate", 
    type=int, default=3)
  parser.add_argument("--yaw", help="Rotation on z in degrees", 
    type=int, default=0)
  parser.add_argument("--pitch", help="Rotation on y in degrees", 
    type=int, default=0)
  parser.add_argument("--roll", help="Rotation on x in degrees", 
    type=int, default=0)
  parser.add_argument("--dz", help="Translation on z", 
    type=int, default=0)
  parser.add_argument("--dy", help="Translation on y", 
    type=int, default=0)
  parser.add_argument("--dx", help="Translation on x", 
    type=int, default=0)
  parser.add_argument("--visualize", 
    help="Reading the input from a file", action="store_true", default=False)
  args = parser.parse_args() 

  # Compute some random points
  # N x 3
  P = GenerateRandom3DPoints(args.n_points)
  # Get rotation matrix
  # 3 x 3
  rot = ComputeRotationMatrix(args.yaw, args.pitch, args.roll)
  t = np.array([args.dx, args.dy, args.dz]).reshape(3, 1)
  print(t.shape)
  # Compute Q = R * P
  # 3 x N
  Q = np.dot(rot, P) + np.tile(t, args.n_points)
  print(f"P\n{P}\nQ\n{Q}\nRot\n{rot}\nt\n{t}")

  rot_est, t_est = ComputeRigidTransformation(P, Q)
  print(f"Rot_est\n{rot_est}\nRot\n{rot}")
  print(f"t_est\n{t_est}\nt\n{t}")

  assert np.allclose(rot, rot_est), "R_truth != R_est"
  assert np.allclose(t, t_est), "t_truth != t_est"

  Q_est = np.dot(rot_est, P) + np.tile(t, args.n_points)

  # Plot stuff 
  if args.visualize:
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot3D(P[0, :], P[1, :], P[2, :], color='gray', marker='o', label='P')
    ax.plot3D(Q[0, :], Q[1, :], Q[2, :], color='green', marker='+', label='Q')
    ax.scatter(Q_est[0, :], Q_est[1, :], Q_est[2, :], color='red', marker='s', label='Qest')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.show()