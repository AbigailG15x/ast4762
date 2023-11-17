#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
from scipy.ndimage import binary_dilation
import hw8_abigailglover_support_functions as D

def apphot(image, star_coords, aperture_radius, sky_annulus_radii):
    """
    Performs aperture photometry on a star within a 2D image.

    Parameters
    ----------
    image : (ndarray)
        A 2D array representing the image containing the star.

    star_coords : (tuple)
        A tuple specifying the (y, x) coordinates of the star in the image.

    aperture_radius : (float)
        The radius of the aperture used for photometry, defining the region
        around the star for flux measurement.

    sky_annulus_radii : (tuple)
        A tuple containing the inner and outer radii of the sky annulus. The
        sky annulus is an annular region surrounding the aperture, used for
        background sky estimation.

    Returns
    -------
    A tuple containing the stellar flux, representing the total flux
    within the aperture, and the average sky, calculated as the mean
    pixel value in the sky annulus. This information is valuable for
    quantitative analysis of stellar brightness within the image.
        
    stellar_flux : (float)
        The total flux within the aperture, representing the brightness
        of the star.

    average_sky : (float)
        The average pixel value in the sky annulus, providing an
        estimate of the background sky level. This value is crucial for
        accurate subtraction of background noise from the stellar flux,
        enabling precise photometric measurements.
    """
    
    # (a) Cut out a sub-image around the approximate center of the star
    cy, cx = map(int, star_coords)
    subimage_size = int(max(aperture_radius, max(sky_annulus_radii)))
    subimage = image[cy - subimage_size//2 : cy + subimage_size//2 + 1,
                     cx - subimage_size//2 : cx + subimage_size//2 + 1]
    print(subimage)
    
    if np.sum(subimage) == 0:
        # No star present, cut the sub-image around the sky annulus
        aperture_subimage = image[cy - subimage_size//2 : cy + subimage_size//2 + 1,
                          cx - subimage_size//2 : cx + subimage_size//2 + 1]

    print(subimage)
    
   # (b) Use disk twice to make masks for the sky annulus and photometry aperture
    apmask = D.disk(aperture_radius, (subimage_size//2, subimage_size//2), subimage.shape)
    skymask = D.disk(sky_annulus_radii[1], (subimage_size//2, subimage_size//2), subimage.shape) \
          ^ D.disk(sky_annulus_radii[0], (subimage_size//2, subimage_size//2), subimage.shape)
    print(apmask)
    print(skymask)
    
    # (c) Calculate the average sky pixel in the annulus
    Nstar = np.sum(apmask)
    Nsky = np.sum(skymask)
    print(Nstar)
    print(Nsky)
    
    # Check if Nsky is non-zero before calculating anavg
    if Nsky != 0:
        apraw = np.sum(subimage[apmask])
        anavg = np.sum(subimage[skymask]) / Nsky
    else:
        anavg = 1.0

    # (d) Subtract the average sky from each pixel in the sub-image
    subimage_copy = subimage.copy()
    subimage_copy -= anavg
    print(subimage)

    # (e) Use disk to make a mask for the photometry aperture
    apmask_dilated = binary_dilation(apmask, iterations=1)
    
    # (f) Calculate the total flux in the aperture
    indices = np.where(apmask_dilated)
    stellar_flux = np.sum(subimage[indices])
    print(indices)
    
    # (g) Return a tuple containing the stellar flux and the average sky
    return stellar_flux, anavg


# In[ ]:




