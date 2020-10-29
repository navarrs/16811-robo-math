# @author     Ingrid Navarro
# @andrewid   ingridn
# @date       Oct 24, 2020
#
# @brief      Hw4 - Q1 Differential Equations
# ------------------------------------------------------------------------------

#
# Includes ---------------------------------------------------------------------
import matplotlib.pyplot as plt
import os 
import numpy as np 

OUTDIR = "../out/q2"
if not os.path.exists(OUTDIR):
  os.makedirs(OUTDIR)

#
# Helper Methods ---------------------------------------------------------------
def f(x, y):
  return x**3 + y**3 - 2*x**2 + 3*y**2 - 8

def Plot(title = 'Iso-contours f(x)'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Iso contours
    x = np.linspace(-0.5, 1.5, num=50)
    y = np.linspace(-2.5, 0.5, num=50)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    plt.contourf(X, Y, Z, 20, cmap='RdGy')
    plt.colorbar()
    
    # x_ = np.linspace(-3, 3, num=20)
    # y_ = np.linspace(-3, 3, num=20)
    # X_, Y_ = np.meshgrid(x_, y_)
    # dx, dy = np.gradient(f(X_, Y_))
    # ax.quiver(X_, Y_, dx, dy)
  
    
    # Plot critical points 
    x_critical = [0, 0, 4/3, 4/3]
    y_critical = [0, -2, 0, -2]
    plt.scatter(x_critical, y_critical, c='k', s=20, label='critical points')
    
    for i in range(len(y_critical)):
      ax.annotate(f"({np.round(x_critical[i], 2)}, {np.round(y_critical[i], 2)})", 
                  xy=(x_critical[i], y_critical[i]+0.06), fontsize=8)
    
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(OUTDIR, "iso-contours.png"))
    plt.show()



# #
# # Main program -----------------------------------------------------------------
if __name__ == "__main__":

    
    Plot()