#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from astropy.io import fits
'''
    Generates a 2D mask array with a disk image.

    This function creates a 2D mask array with 
    True values for each pixel whose center is 
    within a specified radius from the given 
    position and False elsewhere.

    Parameters
    ----------
    radius: The radius of the disk.
    
    center: A 2-element tuple specifying the 
    center of the disk as (y, x).
    
    shape: A 2-element tuple specifying the 
    shape of the output image as (y, x).

    Returns
    ----------
    mask: A 2D mask array representing the disk 
    image, where True pixels are inside the disk, 
    and False pixels are outside.
'''

def disk(radius, center, shape):
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((y - center[0])**2 + (x - center[1])**2)
    mask = dist_from_center <= radius
    return mask

# Define parameters
radius = 6.2
center = (12.3, 14.5)
image_shape = (25, 30)

# Generate the disk mask
mask = disk(radius, center, image_shape)

# Store the mask in a FITS file
fits.writeto("disktest.fits", mask.astype(int), overwrite=True)

# Print the values of pixel indices [14, 14] and [2, 1]
print("Pixel value at [14, 14]:", mask[14, 14])
print("Pixel value at [2, 1]:", mask[2, 1])


# In[ ]:




