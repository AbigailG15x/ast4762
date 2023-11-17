#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Abigail Glover
# ast4762 - HW 9
# 11/12/2023


# In[64]:


# Homework 8 Solution

# UCF AST5765/4762 HW8 Solution
# Joseph Harrington <jh@physics.ucf.edu>

# Revision history:
# Original 2007-10-21 by jh
# Updated  2008-10-28 by Kevin Stevenson
# Updated  2009-11-23 by jh
# Updated  2010-10-26 by jh
# Updated  2014-10-07 by jh, py3
# Updated  2018-10-30 by jh, print prob numbers
# Updated  2023-11-16 by Abigail Glover


from hw7_sol import *
import hw8_abigailglover_support_functions
import gaussian as gs
import time
import getpass

#import pyds9
#ds9Win = pyds9.DS9()
#ds9Win.set_np2arr(obj_data[0])
import rdpharo_win as rp

# === Problem 2 ===

print("=== Problem 2 ===")

# Hard-coded values
folder_path = "hw6_data"

# Define lampon and lampoff files
lampon_prefix = "k_lampon_"
lampoff_prefix = "k_lampoff_"

# List all files in the folder
all_files = os.listdir(folder_path)

# Filter lampon and lampoff files
lampon_files = [file.replace(".fits", "") for file in all_files if file.startswith(lampon_prefix)]
lampoff_files = [file.replace(".fits", "") for file in all_files if file.startswith(lampoff_prefix)]

# Sort the files
lampon_files.sort()
lampoff_files.sort()

# Convert lists to NumPy arrays
lamponfile = np.array(lampon_files)
lampofffile = np.array(lampoff_files)

# Print the resulting arrays
print("lamponfile:", lamponfile)
print("lampofffile:", lampofffile)

#lamponfile = np.array([
#	'k_lampon_1',
#	'k_lampon_2',
#	'k_lampon_3',
#	'k_lampon_4',
#	'k_lampon_5'
#])
#lampofffile = np.array([
#	'k_lampoff_1',
#	'k_lampoff_2',
#	'k_lampoff_3',
#	'k_lampoff_4',
#	'k_lampoff_5'
#])

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
  print(lamponfile[k])

# lampoff data
for k in np.arange(nlampoff):
  infile                      = datadir + lampofffile[k] + fext
  lampoffhead, lampoffdata[k] = rp.rdpharo(infile)
  print(lampofffile[k])
          
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
username = getpass.getuser()

flathead.add_history('Flat field made '
                    + time.strftime('%a %b %d %H:%M:%S %Z %Y')
                    + ' ' + username)
flathead.add_history('Normalized difference of medians of lamp off and lamp on data.')
flathead.add_history('Some header entries (like times) are wrong.')

fits.writeto('flat.fits', np.float32(flatdata), flathead, overwrite=True,
             output_verify='silentfix') # due to bug in astropy.io.fits

# Define prtpix
prtpix = (217, 184)

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

# FIXED 16 Nov 2023
# Convert object type
obj_data = obj_data.astype(np.float64)

# Carry out the division
obj_data /= flatdata                     # broadcasting is wonderful!

# FIXED 16 Nov 2023
# Updated print statement
print("obj_data[217, 184] = " + str(obj_data[prtpix[0], prtpix[1]]))


objhead.add_history(  'Flattened '
                    + time.strftime('%a %b %d %H:%M:%S %Z %Y')
                    + ' ' + getpass.getuser())

fits.writeto(objfile[-1]+'_flat'+fext, np.float32(obj_data[-1]), objhead,
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

# FIXED 16 Nov 2023
# Input data from above chart into the following

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
    gheight  = objects_array[frame, gcy, gcx]                    # guess height
    im = objects_array[frame, gcy-di:gcy+di+1, gcx-di:gcx+di+1]  # extract star box
    (width, center, height, err) = gs.fitgaussian(im - np.median(im),
                                                  x=None,  # use default indices
                                                  guess=(gwidth,
                                                         gcenter,
                                                         gheight))
    photometry[star, frame, iwidth]     = np.mean(width)
    photometry[star, frame, [icy, icx]] =  np.array(center) \
                                         + np.array((gcx, gcx)) - di
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


# In[121]:


########################### HW 9 START #####################################

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import hw9_abigailglover_support_functions as hw9_SF
import pandas as pd
import hw8_abigailglover_support_functions as D
import gaussian as g  

# Print problem statement
print("\n\n\nHomework 9")
print("Problem 3\n")

# Generate a noiseless array
image_size = 21
sigma = 2.0
center = (image_size-1) / 2

# Create a meshgrid
x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))

# Generate a 2D Gaussian
gaussian = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))

# Normalize the Gaussian
gaussian /= np.sum(gaussian)

image = gaussian / np.sum(gaussian)

guess = ((sigma, sigma), (center, center), image.max())
(fw, fc, fh, fe) = g.fitgaussian(image - np.median(image), guess=guess)

# Set aperture, sky inner and outer radii
aprad = 6.0 
skyin = 7.0
skyout = 10.0

# Plot the normalized Gaussian
plt.imshow(gaussian, cmap='viridis', origin='lower', extent=[0, image_size, 0, image_size])
plt.title('Normalized Gaussian')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(0, image_size)
plt.ylim(0, image_size)
plt.colorbar(label='Normalized Intensity')
plt.show()

# Plot Aperture
plt.imshow(apmask, cmap='viridis', origin='lower', extent=[0, image_size, 0, image_size])
plt.title('Aperture Mask')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(0, image_size)
plt.ylim(0, image_size)
plt.show()

# Plot Sky Annulus
plt.imshow(skymask, cmap='viridis', origin='lower', extent=[0, image_size, 0, image_size])
plt.title('Sky Annulus Mask')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(0, image_size)
plt.ylim(0, image_size)
plt.show()

# Run apphot on the generated image
star_coords = (center, center)
result = hw9_SF.apphot(image, star_coords, aprad, (skyin, skyout))

# Extract stellar flux and average sky from the result
stellar_flux, anavg = result

# Print the result
print("Stellar Flux:", stellar_flux)
print("Average Sky:", anavg)

print("\nThe values that should have printed are:\
\nStellar Flux: ~1\
\nAverage Sky: ~0\
\nHowever, my code is printing out NaN values,\
\nmeaning that somewhere in my main file or function \
\nthere could be a division by zero, square root of a\
\nnegative, or improper masking that I am missing. Several\
\nmethods of (unsuccessful) debugging are noted in my log.")


# In[81]:


# Print problem statement
print("\nProblem 4\n\n")

# Use the gaussian to fit the data to a star box
def fit_star_box(im, gwidth, gcenter, gheight):
    (width, center, height, err) = gs.fitgaussian(im - np.median(im),
                                                  x=None,
                                                  guess=(gwidth, gcenter, gheight))
    return np.mean(width), np.array(center)

# Write the dophot routine
def dophot(table, data_array, photometry):
    results = []

    # Go through the rows in the table
    for index, row in table.iterrows():
        star_coords = (row['yg'], row['xg'])  # Assuming these are the column names in your table
        aperture_radius = 3.0 * row['width']  # Adjust as needed
        sky_annulus_radii = (5.0 * row['width'], 8.0 * row['width'])  # Adjust as needed

        # Extract star box from the data array
        gcy, gcx = map(int, star_coords)  # Convert to integers
        di = 5  # Half-width of the star box
        star_box = data_array[0, gcy - di:gcy + di + 1, gcx - di:gcx + di + 1]

        # Fit Gaussian to the star box
        gwidth = (1.0, 1.0)  # Initial guess for Gaussian width
        gcenter = np.array((di, di))  # Initial guess for Gaussian center
        gheight = np.median(star_box)  # Initial guess for Gaussian height
        width, center = fit_star_box(star_box, gwidth, gcenter, gheight)

        # Run apphot
        result = hw9_SF.apphot(data_array[index], star_coords, aperture_radius, sky_annulus_radii)

        # Append results to a list
        results.append(result)

    # Convert the list of results to a new DataFrame
    results_df = pd.DataFrame(results, columns=['StellarFlux', 'AverageSky'])

    # Concatenate the original table with the new results
    result_table = pd.concat([table, results_df], axis=1)

    return result_table

# Example usage
table_hw8 = pd.DataFrame(photometry.reshape(-1, photometry.shape[-1]), columns=['yg', 'xg', 'width', 'cy', 'cx', 'star', 'sky'])
resulting_table = dophot(table_hw8, objects_array, photometry)

# Print the resulting table to an ASCII text file
resulting_table.to_csv('resulting_table_hw9.txt', sep='\t', index=False)


# In[ ]:




