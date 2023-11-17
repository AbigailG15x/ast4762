#! /usr/bin/env python3

# UCF AST5765/4762 HW8 Solution
# Joseph Harrington <jh@physics.ucf.edu>

# Revision history:
# Original 2007-10-21 by jh
# Updated  2008-10-28 by Kevin Stevenson
# Updated  2009-11-23 by jh
# Updated  2010-10-26 by jh
# Updated  2014-10-07 by jh, py3
# Updated  2018-10-30 by jh, print prob numbers


from hw7_abigailglover import *
import hw8_abigailglover_support_functions
import gaussian as gs

import pyds9
ds9Win = pyds9.DS9()
ds9Win.set_np2arr(objdata[0])

# === Problem 2 ===

print("=== Problem 2 ===")

# hard-coded values
# hard-coded values
folder_path = "hw6_data"

# Define lampon and lampoff files
lampon_prefix = "k_lampon_"
lampoff_prefix = "k_lampoff_"

# List all files in the folder
all_files = os.listdir(folder_path)

# Filter lampon and lampoff files
lampon_files = [file for file in all_files if file.startswith(lampon_prefix)]
lampoff_files = [file for file in all_files if file.startswith(lampoff_prefix)]

# Sort the files
lampon_files.sort()
lampoff_files.sort()

# Convert lists to NumPy arrays
lamponfile = np.array(lampon_files)
lampofffile = np.array(lampoff_files)

# query data for sizes
nlampon  = lamponfile.size
nlampoff = lampofffile.size

# Allocate data arrays
lampondata   = np.zeros((nlampon,  ny, nx))
lampoffdata  = np.zeros((nlampoff, ny, nx))

# lampon data
for k in np.arange(nlampon):
  infile                    = datadir + lamponfile[k] + fext
  lamponhead, lampondata[k] = rp.rdpharo(infile)

# lampoff data
for k in np.arange(nlampoff):
  infile                      = datadir + lampofffile[k] + fext
  lampoffhead, lampoffdata[k] = rp.rdpharo(infile)

# calculate the difference of the medians
lampon  = np.median(lampondata,  axis=0)
lampoff = np.median(lampoffdata, axis=0)
flatdata = lampon - lampoff

# normalize, get away from the edges, particularly the junk in the upper right
norm = np.median(flatdata[normregion[0][0] : normregion[1][0],
                          normregion[0][1] : normregion[1][1]])
flatdata /= norm

# save
flathead = lampoffhead.copy()
flathead.add_history(  'Flat field made '
                    + time.strftime('%a %b %d %H:%M:%S %Z %Y')
                    + ' ' + pwd.getpwuid(os.getuid())[0])
flathead.add_history('Normalized difference of medians of lamp off and lamp on data.')
flathead.add_history('Some header entries (like times) are wrong.')

fits.writeto('flat.fits', np.float32(flatdata), flathead, overwrite=True,
             output_verify='silentfix') # due to bug in astropy.io.fits

print("flatdata[217, 184] = "+str(flatdata[prtpix[0], prtpix[1]]))


#2b

# flatten
# This "fixes" the flat field so that there is no division by 0.  The
# value of 1 that we use here is wrong, but so is the 0...and so is
# the value in the object frames for those pixels.  We are not masking
# bad pixels in this exercise, but if we were, we would mark all the
# pixels with 0 in the flat as bad, and would not use them or would
# interpolate to replace them.
bad = np.where(flatdata == 0)
flatdata[bad] = 1.
objdata /= flatdata                     # broadcasting is wonderful!

print("objdata[:, 217, 184] = "+str(objdata[:, prtpix[0], prtpix[1]]))

objhead.add_history(  'Flattened '
                    + time.strftime('%a %b %d %H:%M:%S %Z %Y')
                    + ' ' + pwd.getpwuid(os.getuid())[0])

fits.writeto(objfile[-1]+'_flat'+fext, np.float32(objdata[-1]), objhead,
             overwrite=True,
             output_verify='silentfix') # due to bug in astropy.io.fits




# === Problem 3 ===

print("=== Problem 3 ===")

# see disk.py
disktest = hw8_abigailglover_support_functions.disk(6.2, (12.3, 14.5), (25, 30))
fits.writeto('disktest.fits', np.uint8(disktest), overwrite=True,
             output_verify='silentfix') # due to bug in astropy.io.fits

# print pixel values requested in the problem
d1y, d1x = [14, 14]
d2y, d2x = [ 2,  1]
print("disktest["+str(d1y)+", "+str(d1x)+"] = "+str(disktest[d1y, d1x]))
print("disktest["+str(d2y)+", "+str(d2x)+"] = "+str(disktest[d2y, d2x]))



print("=== Problem 4 ===")

print("Problem 4")

# Stellar photometry:
photometry = np.array(
  [
#  yguess, xguess,     width,     cy,     cx,     star,     sky
  # star 0
   [[   698,    512,    0.959,    698.61,   512.39,   np.nan,  np.nan],  # frame 0
    [   464,    517,    0.931,    464.37,   517.42,   np.nan,  np.nan],  # frame 1
    [   228,    522,    1.070,    228.43,   521.93,   np.nan,  np.nan]], # frame 2
  # star 1
   [[   668,    520,    1.026,    667.88,   520.23,   np.nan,  np.nan],  # frame 0
    [   434,    525,    0.961,    433.58,   525.30,   np.nan,  np.nan],  # frame 1
    [   198,    530,    1.144,    197.64,   529.79,   np.nan,  np.nan]], # frame 2
  # star 2
   [[   568,    283,    1.063,    568.26,   283.30,   np.nan,  np.nan],  # frame 0
    [   334,    288,    0.955,    333.96,   288.36,   np.nan,  np.nan],  # frame 1
    [    98,    293,    1.071,    98.07,    292.90,   np.nan,  np.nan]]  # frame 2
  ], dtype=float)

# These are symbolic names to use as indices in the photometry array.
(iyg, ixg, iwidth, icy, icx, istar, isky) = np.arange(photometry.shape[2])
(nstar, nframe, npar) = photometry.shape

# Fill in the missing yg and xg values in the photometry table
for frame in np.arange(1, nframe):
  # Note how breaking the next line shows the math clearly:
  offset =  photometry[0, frame, [iyg, ixg]] \
          - photometry[0,     0, [iyg, ixg]]
  for star in np.arange(1, nstar):
    photometry[star, frame, [iyg, ixg]] =  photometry[star, 0, [iyg, ixg]] \
                                         + offset



# === Problem 5 ===

print("=== Problem 5 ===")

#5a

gwidth  = (1., 1.)                       # guess Gaussian width in x and y
di      = 5                              # half-width of star box; int for index
gcenter = (float(di), float(di))         # guess center (float) in star box
print('# star frame yg   xg   width  cy      cx')
for   star  in np.arange(nstar):
  for frame in np.arange(nframe):
    # guess center index, convert to integers for indexing, unpack into scalars
    gcy, gcx = np.array(photometry[star, frame, [iyg, ixg]], dtype=int)
    gheight  = objdata[frame, gcy, gcx]                    # guess height
    im = objdata[frame, gcy-di:gcy+di+1, gcx-di:gcx+di+1]  # extract star box
    (width, center, height, err) = gs.fitgaussian(im - np.median(im),
                                                  x=None,  # use default indices
                                                  guess=(gwidth,
                                                         gcenter,
                                                         gheight))
    photometry[star, frame, iwidth]     = np.mean(width)
    photometry[star, frame, [icy, icx]] =  np.array(center) \
                                         + np.array((gcy, gcx)) - di
    print('# %d    %d     %3.0f  %3.0f  %.3f  %6.2f  %6.2f'
           % ((star, frame) + tuple(photometry[star, frame,:-2])))



# star frame yg   xg   width  cy      cx
# 0    0     698  512  0.959  698.61  512.39
# 0    1     464  517  0.931  464.37  517.42
# 0    2     228  522  1.070  228.43  521.93
# 1    0     668  520  1.026  667.88  520.23
# 1    1     434  525  0.961  433.58  525.30
# 1    2     198  530  1.144  197.64  529.79
# 2    0     568  283  1.063  568.26  283.30
# 2    1     334  288  0.955  333.96  288.36
# 2    2      98  293  1.071   98.07  292.90


#5b


width  = np.mean(photometry[:,:,iwidth])
aper   = 3. * width  # this should enclose almost all flux
skyin  = 5. * width  # far enough out not to get any significant flux
skyout = 8. * width  # enclose several times more pixels than in aperture

subsize = (skyout + 1) * 2 + 1  # this is more than enough to enclose
                                # all sky if centered on the star

print("The mean width               is %.1f pixels." % (width))
print("The aperture radius          is %.1f pixels." % (aper))
print("The inner sky annulus radius is %.1f pixels." % (skyin))
print("The outer sky annulus radius is %.1f pixels." % (skyout))
print("The subimage size            is %.1f pixels." % (subsize))
