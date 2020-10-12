# @author     Ingrid Navarro
# @andrewid   ingridn
# @date       Oct 9, 2020
#
# @brief      Hw3 - Q4 Plane fitting

#
# Includes ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import q2

#
# Problem implementation -------------------------------------------------------
def EstimatePlane(P):
  # not normalized
  b = np.ones(P.shape[0], dtype=np.float)
  n = q2.Solve(P, b)
  d = np.mean(abs(b - np.dot(P, n))/np.linalg.norm(n))
  
  # normalized
  c = np.mean(P, axis=0, dtype=np.float)
  P_c = P - c
  Pc = P_c.T @ P_c
  U, S, V_T = scipy.linalg.svd(Pc)
  n_ = U[:, -1]
  d_ = np.mean(abs(np.dot(P, n_))/np.linalg.norm(n_))
  print(np.mean(abs((1 - (np.dot(P, n_) + d_))/np.linalg.norm(n_))))
  # avg_d = np.mean(abs(1 - (np.dot(P, n_) + d_))/np.linalg.norm(n_))
  # print(avg_d)
  # Denormalize n
  # n = n_ / d_
  # d = np.mean(abs(1 - np.dot(P, n))/np.linalg.norm(n))
  
  n = n_
  d = d_
  return n, d

#
# Helper methods ---------------------------------------------------------------
def Plot(points, plane = None, d = 1, title="Point Cloud", with_limits=True):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  
  x, y, z = points[:, 0], points[:, 1], points[:, 2]
  minx, maxx = np.min(x), np.max(x)
  miny, maxy = np.min(y), np.max(y)
  minz, maxz = np.min(z), np.max(z)

  # PLOT POINT CLOUD
  # X - positive right
  # Y - positive down
  # Z - positive forward
  ax.set_xlabel('x')
  ax.set_ylabel('z')
  ax.set_zlabel('y')
  
  ax.scatter(x, z, y, color='r')
  if with_limits:
    ax.set_xlim(minx, maxx)
    ax.set_ylim(minz, maxz)
    ax.set_zlim(maxy, miny)
    
  # PLOT PLANE
  if np.any(plane) != None:
    x = np.linspace(minx, maxx, 5)
    y = np.linspace(miny, maxy, 5)
    X, Y = np.meshgrid(x, y)
    Z = (d - plane[0]*X - plane[1]*Y) / plane[2]
    ax.plot_surface(X, Z, Y, alpha=0.3)
  
  plt.show()
  
#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":

    # Q4.A
    print(f"\n----------------------------------------------------------------")
    print(f"Question 4(a) - clear_table.txt")
    clear_table = np.loadtxt("../data/clear_table.txt", dtype=np.float)
    plane, avg_d = EstimatePlane(clear_table)
    print(f"Plane normal: {plane} Av. distance {avg_d}")
    Plot(clear_table, plane, avg_d)
    
    # Q4.B
    print(f"\n----------------------------------------------------------------")
    print(f"Question 4(b) - cluttered_table.txt")
    cluttered_table = np.loadtxt("../data/cluttered_table.txt", dtype=np.float)
    plane, avg_d = EstimatePlane(cluttered_table)
    print(f"Plane normal: {plane} Av. distance {avg_d}")
    Plot(cluttered_table, plane, avg_d)
    
    # Q1.C
    