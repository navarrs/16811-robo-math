# @author     Ingrid Navarro
# @andrewid   ingridn
# @date       Oct 24, 2020
#
# @brief      Hw4 - Q1 Differential Equations
# ------------------------------------------------------------------------------

#
# Includes ---------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os 

OUTDIR = "../out/q1"
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

print_tables = True

if print_tables:
    from prettytable import PrettyTable

#
# Problem implementation -------------------------------------------------------
# Q1.A
def dy(y):
    if not np.isclose(y, 0.0):
        return 1/(3 * y**2)
    
def Exacty(x):
    return x ** (1/3)

# Q1.B
def EulersMethod(y0, f, n, h=0.05):
    """ Euler's Method to solve numerical differential equations. """
    def Step(y):
        return y + h * f(y)

    y_euler = [y0]
    for i in range(n-1):
        y_euler.append(Step(y_euler[i]))
        
    return np.asarray(y_euler)

# Q1.C
def RungeKutta4(y0, f, n, h=0.05):
    """ Runge-Kutta Method of order 4 to solve numerical differential 
        equations. """
    def K1(y):
        """ Estimates y' at starting point. """
        return h * f(y)
    
    def K2(y):
        """ Estimates y' at the midpoint between xn + xn+1 using a y-value
            obtained with K1. """
        y_ = y + K1(y) / 2
        return h * f(y_)
    
    def K3(y):
        """ Estimates y' at the midpoint between (xn + xn+1)/2 using a y-value
            obtained with K2. """
        y_ = y + K2(y) / 2
        return h * f(y_)

    def K4(y):
        """ Estimates y' at the point xn+1 using a y-value obtained by advancing
            from yn by K3. """
        y_ = y + K3(y)
        return h * f(y_)
        
    y_rk4 = [y0]
    for i in range(n-1):
        yn = y_rk4[i]
        yn1 = yn + (K1(yn) + 2*K2(yn) + 2*K3(yn) + K4(yn)) / 6
        y_rk4.append(yn1)
    
    return np.asarray(y_rk4)
    
# Q1.D
def AdamsBashforth4(yn, yn_1, yn_2, yn_3, f, n, h=0.05):
    """ Adams-Bashforth Method of order 4 to solve numerical differential 
        equations. """
    def Step():
        step = 55 * f(yn) - 59 * f(yn_1) + 37 * f(yn_2) - 9 * f(yn_3)
        return yn + h * step /24
    
    y_ab4 = [yn_3, yn_2, yn_1, yn]
    for i in range(n-1):
        y_ab4.append(Step())
        yn = y_ab4[i+4]
        yn_1 = y_ab4[i+3]
        yn_2 = y_ab4[i+2]
        yn_3 = y_ab4[i+1]
    
    return np.asarray(y_ab4)[3:][::-1]

#
# Helper Methods ---------------------------------------------------------------


def Plot(x, y1, y2, y3, y4, title = 'Analytical vs Numerical solutions'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    
    plt.plot(x, y1, color='r', label='Exact')
    plt.plot(x, y2, color='b', label='Euler')
    plt.plot(x, y3, color='g', label='RK4')
    plt.plot(x, y4, color='m', label='AB4')
    
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(OUTDIR, "ode-solutions.png"))
    plt.show()

#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":

    h = 0.05
    interval = [0.0, 1.0]
    print(f"\n----------------------------------------------------------------")
    print("Question 1 (a) Exact y(x) solution")
    x = np.arange(start=interval[0], stop=interval[1]+h, step=h)
    n = len(x)
    y = Exacty(x)
    
    print(f"\n----------------------------------------------------------------")
    print("Question 1 (b) Euler method")
    y_euler = EulersMethod(1, dy, n, -h)[::-1]
    err_euler = abs(y - y_euler)
    
    if print_tables:
        Teuler = PrettyTable()
        Teuler.add_column('x', x)
        Teuler.add_column('y_Exacty', y)
        Teuler.add_column('y_euler', y_euler)
        Teuler.add_column('error', err_euler)
        Teuler.float_format = ".10"
        print(Teuler)
        
    table = np.zeros((len(x), 4))
    table[:, 0] = x
    table[:, 1] = y
    table[:, 2] = y_euler
    table[:, 3] = err_euler
    np.savetxt(os.path.join(OUTDIR, 'table_euler.csv'), 
               table, fmt='%10.5f', delimiter=',')
    
    print(f"\n----------------------------------------------------------------")
    print("Question 1 (c) Runge-Kutta 4")
    y_rk4 = RungeKutta4(1, dy, n, -h)[::-1]
    err_rk4 = abs(y - y_rk4)
    
    if print_tables:
        Trk4 = PrettyTable()
        Trk4.add_column('x', x)
        Trk4.add_column('y_Exacty', y)
        Trk4.add_column('y_rk4', y_rk4)
        Trk4.add_column('error', err_rk4)
        Trk4.float_format = ".10"
        print(Trk4)
        
    table[:, 0] = x
    table[:, 1] = y
    table[:, 2] = y_rk4
    table[:, 3] = err_rk4
    np.savetxt(os.path.join(OUTDIR, 'table_rk4.csv'), 
               table, fmt='%10.5f', delimiter=',')

    
    print(f"\n----------------------------------------------------------------")
    print("Question 1 (d) Adams-Bashforth 4")
    y0, y1, y2, y3 = Exacty(1), Exacty(1.05), Exacty(1.10), Exacty(1.15)
    print(f"Starting with y0: {y0}, y1: {y1}, y2: {y2}, y3: {y3}")
    y_ab4 = AdamsBashforth4(y0, y1, y2, y3, dy, n, -h)
    err_ab4 = abs(y - y_ab4)
    
    if print_tables:
        Tab4 = PrettyTable()
        Tab4.add_column('x', x)
        Tab4.add_column('y_Exacty', y)
        Tab4.add_column('y_ab4', y_ab4)
        Tab4.add_column('error', err_ab4)
        Tab4.float_format = ".10"
        print(Tab4)
    
    table[:, 0] = x
    table[:, 1] = y
    table[:, 2] = y_ab4
    table[:, 3] = err_ab4
    np.savetxt(os.path.join(OUTDIR, 'table_ab4.csv'), 
               table, fmt='%10.5f', delimiter=',')
 
    
    title = "Analytical vs Numerical solutions to y(x)"
    
    # Full table
    table = np.zeros((len(x), 8))
    table[:, 0] = x
    table[:, 1] = y
    table[:, 2] = y_euler
    table[:, 3] = err_euler
    table[:, 4] = y_rk4
    table[:, 5] = err_rk4
    table[:, 6] = y_ab4
    table[:, 7] = err_ab4
    np.savetxt(os.path.join(OUTDIR, 'full_table.csv'), 
               table, fmt='%10.5f', delimiter=',')
    
    Plot(x, y, y_euler, y_rk4, y_ab4, title)