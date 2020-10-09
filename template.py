# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 4, 2020
#
# @brief      
# @notes     

#
# Includes ---------------------------------------------------------------------

#
# Problem implementation -------------------------------------------------------

#
# Helper methods ---------------------------------------------------------------

#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # Use it to verify PA == LDU and Ax == b
  parser.add_argument("--do_assert", help="Asserts that PA == LDU and Ax == b", 
    action="store_true", default=False)
  # If testing a sample from file, add --from-file --path /path/to/file
  parser.add_argument("--random", help="Reading the input from a file", 
    action="store_true", default=False)
  args = parser.parse_args()