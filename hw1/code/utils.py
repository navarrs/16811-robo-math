# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 7, 2020
#
# @brief      Utils to solve homework1

#
# Includes ---------------------------------------------------------------------
import numpy as np

#
# Funcition Implementation -----------------------------------------------------
def ReadMatrix(filename, type_mat=float, delim=","):
  return np.loadtxt(filename, dtype=type_mat, delimiter=delim)

#
# End --------------------------------------------------------------------------