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

def ComputePointSet(R, t, n):
  """
    Generates ground truth pointsets P and Q
  """
  # Compute some random points P (3xN)
  P = GenerateRandom3DPoints(n)
  # Find centroid and shift the data 
  pc = np.mean(P, axis=1, dtype=float).reshape(3, 1)
  Pc = P - np.tile(pc, P.shape[1])

  # Compute transformed points Q_truth = R * P (3XN)
  # Q_truth is rotated about the centroid of P (i.e. Pc) then, translated by this
  # centroid values and translation component.
  # @note to self: np.tile is used to sum columns
  Q = np.dot(R, Pc) + np.tile(pc, n) + np.tile(t, n)
  return P, Q

def GetRandomParameters(nmin=3, nmax=10, degmin=-180, degmax=180, tmin=-10, tmax=10):
  """
    Random parameters to test
  """
  n = np.random.randint(low=nmin, high=nmax)
  yaw = np.random.randint(low=degmin, high=degmax)
  pitch = np.random.randint(low=degmin, high=degmax)
  roll = np.random.randint(low=degmin, high=degmax)
  dx = np.random.randint(low=tmin, high=tmax)
  dy = np.random.randint(low=tmin, high=tmax)
  dz = np.random.randint(low=tmin, high=tmax)
  return n, yaw, pitch, roll, dx, dy, dz

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
  t = qc - pc #np.dot(rot, pc)
  tc = qc - np.dot(rot, pc)
  return rot, t, tc

#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  visualize = False
  #
  # Get random parameters ------------------------------------------------------
  n, yaw, pitch, roll, x, y, z = GetRandomParameters()
  print(f"Parameters:\nn: {n} yaw: {yaw}, pitch: {pitch} roll: {roll} dx: {x} dy: {y} dz: {z}")

  #
  # Generate data --------------------------------------------------------------
  # Ground truth transformation R, t
  R_truth = Euler2RotationMatrix(yaw, pitch, roll)
  t_truth = np.array([x, y, z]).reshape(3, 1)
  P, Q_truth = ComputePointSet(R_truth, t_truth, n)
  print(f"P:\n{P}\nQ:\n{Q_truth}")

  #
  # Estimate transformation ----------------------------------------------------
  R_est, t_est, tc = ComputeRigidTransformation(P, Q_truth)
  
  print(f"Rot_est\n{R_est}\nRot_truth\n{R_truth}")
  print(f"t_est\n{t_est}\nt_truth\n{t_truth}")

  assert np.allclose(R_truth, R_est), "R_truth != R_est"
  print("Success R_truth == R_est")
  assert np.allclose(t_truth, t_est), "t_truth != t_est"
  print("Success T_truth == t_est")

  # Plot stuff 
  if visualize:
    import matplotlib.pyplot as plt

    Q_est = np.dot(R_est, P) + np.tile(tc, n)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Transformation of P into Q")
    ax.plot3D(P[0, :], P[1, :], P[2, :], 
      color='gray', marker='o', label='P')
    ax.plot3D(Q_truth[0, :], Q_truth[1, :], Q_truth[2, :], 
      color='green', marker='+', label='Q')
    ax.scatter(Q_est[0, :], Q_est[1, :], Q_est[2, :], 
      color='red', marker='s', label='Qest')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.show()