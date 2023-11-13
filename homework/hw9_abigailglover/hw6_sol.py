#!/usr/bin/env python
# coding: utf-8


# UCF AST5765/4762 HW7 Solutions
# Joseph Harrington <jh@physics.ucf.edu>

# Revision history:
# Updated 20 Oct 2008 by Kevin Stevenson
# Updated  1 Oct 2009 by Joseph Harrington
# Updated 14 Oct 2010 by Joseph Harrington
# Updated  5 Oct 2014 by Joseph Harrington
# Updated 12 Oct 2016 by Joseph Harrington
# Updated 15 Oct 2019 by Joseph Harrington
# Updated Oct 2023 by TK
# Updated 31 Oct 2023 by Abigail Glover

import os
import numpy as np
# Changed to rdpharo_win for windows machine
import rdpharo_win as rp # FIXED 31 Oct 2023
import time
import os

## FIXED 12 Nov 2023
#import pwd
import astropy.io.fits as fits



#set data dir to loop over files instead of hardcoding
# Updated data directory to a different data location relative to this code
datadir = "hw6_data/" # FIXED 31 Oct 2023
fext     = '.fits'



all_files = os.listdir(datadir)
darkfile =[]
objfile  = []

for f in all_files:
    if f[:4] == 'dark':
        darkfile.append( f[:-5] )
    elif f[:4] == 'star':
        objfile.append( f[:-5] )





#check that it has rm the extension correctly
#print( darkfile )



print("datadir      : " + datadir)
print("fext         : " + fext)
print("last objfile : " + objfile[-1])
print("last darkfile: " + darkfile[-1])


# query data for sizes (DO NOT query header!  Data are truth.  "Data"
# is also plural; "datum" is the singular.)
filename        = datadir +darkfile[0] + fext
tmphead, tmpdat = rp.rdpharo(filename)

#print( filename)

ny, nx = tmpdat.shape
#print( tmpdat.shape)
ndark  = len(darkfile)
nobj   =  len(objfile)

print("ny   : " + str(ny))
print("nx   : " + str(nx))
print("nobj : " + str(nobj))
print("ndark: " + str(ndark))



print(tmphead)



# allocate data arrays
objdata  = np.zeros((nobj,  ny, nx), dtype=float)
darkdata = np.zeros((ndark, ny, nx), dtype=float)

print("objdata  size: " + str( len(objdata) ) )
print("darkdata size: " + str( len(darkdata) ) )

# read the data

# object data
# would do "for k in datadir+objfile+fext:", but we need k separately

for k in range(nobj):
  infile              = datadir + objfile[k] + fext
  objhead, objdata[k] = rp.rdpharo(infile)  # NOTE: Implicit type conv.
  # The raw data are 32-bit integers but objdata is 64-bit floats.
  # Note that in the line above, [k] is the same as [k,:,:].  Cool, huh?
  # This lets you write code that works on arrays for which you don't
  # know the number of dimensions, but you know you want to access the
  # first N dimensions some particular way.
  # On each iteration here, the header overwrites objhead, so the last
  # header is the one you get at the end of the loop, as the problem
  # requests.

# dark data
for k in range(ndark):
  infile                = datadir + darkfile[k] + fext
  darkhead, darkdata[k] = rp.rdpharo(infile)  # NOTE: Implicit type conv.

print(" objhead date: " +  objhead["DATE-OBS"])
print("darkhead date: " + darkhead["DATE-OBS"])

print("")
print("========")
print("")

print("Can't use TIME-OBS, all but the first of the data frames are missing it.")
print("LOOK at ALL of your data.  Don't trust anything!")
print("This is not a made-up example.  This is how these frames really arrived.")



print("")
print("======")
print("")

darkmeddata = np.median(darkdata, axis=0)
prty = 217 # y-pixel value to print
prtx = 184 # x-pixel value to print
print("darkmeddata[%d, %d] = %f" % (prty, prtx, darkmeddata[prty, prtx]))
# darkmeddata[217, 184] = 22.000000



print("")
print("======")
print("")

# make the new header
darkhead.add_history(  'medcombine: '
                     + time.strftime('%a %b %d %H:%M:%S %Z %Y')
                     + ' ' + pwd.getpwuid(os.getuid())[0])
darkhead.add_history('Median combination of '+str(darkdata.shape[0])+' images.')
darkhead.add_history('Some header entries (like times) are wrong.')

outname     = darkfile[0][0:-1]
# make the data float32, not float64, to save space
fits.writeto(outname+'med'+fext,
             np.float32(darkmeddata), darkhead, overwrite=True,
             output_verify='silentfix') # due to bug in astropy.io.fits

print(f"See FITS file {outname+'med'+fext}.")


#print("")
#print("======")
#print("")

#print("Before subtraction, objdata[%d, %d, %d] = %f" % (0, prty, prtx, objdata[0, prty, prtx]))
#objdata -= darkmeddata
#print("After  subtraction, objdata[%d, %d, %d] = %f" % (0, prty, prtx, objdata[0, prty, prtx]))

# This is called "broadcasting".  If two arrays of non-matching size
# are to be operated on, and one fits nicely into the other, the
# smaller one will be repeated to make up the empty space.  This is
# like an implicit "for" loop that subtracts darkmeddata from each
# image in objdata in turn, but it's faster and easier to read.

#fits.writeto('mediancombinedsubtraction'+fext,
#             np.float32(objdata[0]), objhead, overwrite=True,
#             output_verify='silentfix') # due to bug in astropy.io.fits
