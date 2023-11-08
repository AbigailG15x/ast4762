#! /usr/bin/env python3

# UCF AST5765/4762
# HW7 Solutions
# Joseph Harrington <jh@physics.ucf.edu>

# Revision history:
# Original 21 Oct 2007 by Joseph Harrington
# Updated  27 Oct 2008 by Kevin Stevenson
# Updated   1 Oct 2009 by Joseph Harrington
# Updated   7 Oct 2014 by Joseph Harrington, python 3
# Updated  06 Nov 2023 by Kenneth Goodis Gordon

## Call all importds at top of file
import matplotlib.pyplot as plt
import numpy as np
from hw6_sol import *                    ## Import for Problem 2
import medcombine as mc                  ## Import for Problems 3 and 5


print("UCF AST 5765/4762")
print("HW 7 Solutions")
print("")

# === Problem 1 ===

print("\n=== Problem 1 ===")
print("")

print('/nFiles and folders created as requested.')
print('/nInitial commits and pushes to GitHub completed.')
print('/nFile begins with requested header info.')


# === Problem 2 ===

print("\n=== Problem 2 ===")
print("")

print('\nAll routines and files from hw6_sol imported correctly. See import ' \
      'at the top of the main homework file.')
print('\nChanges made correctly to hw6_sol with # FIXED added as necessary.')


# === Problem 3 ===

print("\n=== Problem 3 ===")
print("")

print('\nSee routine normmedcomb() in medcombine.py. Import at top of main ' \
      'homework code')
    

# === Problem 4 ===

print("\n=== Problem 4 ===")
print("")

# get away from the edges, particularly the junk in the upper right

normregion = ((225, 225), (-225, -225)) # ((y1, x1), (y2, x2))

# do the normalized median combination
normsky, normfact = mc.normmedcomb(objdata, normregion)

# write results, using 32-bit floats
fits.writeto('sky_13s_mednorm'+fext, np.float32(normsky), objhead,
             overwrite=True,
             output_verify='silentfix') # due to bug in astropy.io.fits

print("Normalization factors:")
print(normfact)
print("")

# [10864. 11057. 10536. 10559. 11046. 10623.  9916.  9916.  9859.]


# === Problem 5 ===

print("\n=== Problem 5 ===")
print("")

print('\nSee routine skycormednorm() in medcombine.py. Import at top of main' \
      ' homework code')

# Apply skycormednorm in a loop to all the object data in place
for k in np.arange(nobj):
    objdata[k] = mc.skycormednorm(objdata[k], normsky, normregion)

# Save the last file.
# Make the data float32, not float64, to save space.
savedata = np.float32(objdata[-1])  # will use twice, so compute only once

fits.writeto(objfile[-1]+'_nosky'+fext, savedata, objhead, overwrite=True,
             output_verify='silentfix') # due to bug in astropy.io.fits

print("\nThe value of pixel [217, 184] is "+savedata[217, 184])


# === Problem 6 ===

print("\n=== Problem 6 ===")
print("")

print('/nLog file is completed, including info on wrapping up.')
print('/nFinal commits and pushes to GitHub completed and screenshot made.')
print('/nHomework 7 completed and submitted to Webcourses.')
