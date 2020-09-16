# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 7, 2020
#
# @brief      Utils to solve homework1

#
# Includes ---------------------------------------------------------------------
import array_to_latex as a2l
import numpy as np
#
# Funcition Implementation -----------------------------------------------------
def ReadMatrix(filename, type_mat=float, delim=","):
  return np.loadtxt(filename, dtype=type_mat, delimiter=delim)

def LatexArr(A, frmt='{:6.3f}', arrtype='array'):
  return a2l.to_ltx(A, frmt = frmt, arraytype = arrtype)
#
# End --------------------------------------------------------------------------