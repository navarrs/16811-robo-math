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
import random
from enum import Enum

class PLANE(Enum):
  LZ = 0
  HZ = 1
  LX = 2
  HX = 3

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
  avg_d = np.mean(abs((1 - (np.dot(P, n_) + d_))/np.linalg.norm(n_)))
    
  n = n_
  d = d_
  return n, d, avg_d

def RANSAC(points, n_samples = 10, n_iters = 100, dthresh = 0.015):
  N = len(points)
  
  best_ninliers = 0
  best_n = 0
  best_d = 0
  best_avg_d = 0
  for i in range(n_iters):
    # Get sample points
    samples = random.sample(range(N), n_samples)
    testp = points[samples]
    
    # Estimate a plane from sample points
    n, d, avg_d = EstimatePlane(testp)
    # print(n, d)
    
    # Count the number of inliers
    inliers = 0
    for j in range(N):
      p = points[j]
      d_ = np.mean(abs((1 - (np.dot(p, n) + d))/np.linalg.norm(n)))
      if d_ <= dthresh:
        inliers += 1
    
    if inliers > best_ninliers:
      best_ninliers = inliers
      best_n = n
      best_d = d
      best_avg_d = avg_d
  
  print(f"Best model n: {best_n}, d: {best_d}, av. d: {best_avg_d} total inliers {best_ninliers}")
  return best_n, best_d, best_avg_d

def MultiPlane(points, n_samples = 10, n_iters = 100, dthresh = 0.015):
  
  
  
  

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
    n, d, avg_d = EstimatePlane(clear_table)
    print(f"Plane normal: {n} Distance: {d}, Av. dist: {avg_d}")
    # Plot(clear_table, plane, avg_d, "Clear Table Point Cloud")
    
    # Q4.B
    print(f"\n----------------------------------------------------------------")
    print(f"Question 4(b) - cluttered_table.txt")
    cluttered_table = np.loadtxt("../data/cluttered_table.txt", dtype=np.float)
    n, d, avg_d = EstimatePlane(cluttered_table)
    print(f"Plane normal: {n} Distance: {d}, Av. dist: {avg_d}")
    # Plot(cluttered_table, plane, avg_d, "Cluttered Table Point Cloud")
    
    # Q1.C
    print(f"\n----------------------------------------------------------------")
    print(f"Question 4(c) - cluttered_table.txt")
    # n, d, avg_d = RANSAC(cluttered_table, 10, 10)
    # Plot(cluttered_table, n, d, "Cluttered Table Point Cloud")
    print(f"Best model normal: {n} Distance: {d}, Av. dist: {avg_d}")
    
    # Q1.D
    print(f"\n----------------------------------------------------------------")
    print(f"Question 4(d) - clean_hallway.txt")
    clean_hallway = np.loadtxt("../data/clean_hallway.txt", dtype=np.float)
    # n, d, avg_d = RANSAC(clean_hallway, 10, 100)
    Plot(clean_hallway) #, n, d, "Cluttered Table Point Cloud")
    # print(f"Best model normal: {n} Distance: {d}, Av. dist: {avg_d}")