#!/usr/bin/env python
# coding: utf-8

# In[21]:


"""
This function calculates the absolute magnitude of
a star and its associated error.

The function calculates the absolute magnitude of 
one or more stars based on their apparent magnitudes,
distances to Earth, and associated errors in apparent
magnitude and distance. It also calculates the error 
in the absolute magnitude.

Parameters
-------------------
apparent_magnitude: Apparent magnitude(s) of the star(s).

distance: Distance(s) to the Earth in parsecs.

apparent_mag_error: Error in apparent magnitude(s).

distance_error: Error in distance(s) in parsecs.

Returns
-------------------
absolute_mag: Absolute magnitude(s) of the star(s).

absolute_mag_error: Error in absolute magnitude(s).

References
-------------------
. . [1] Downy, A., 2015, "Think Python: How to Think Like a Computer Scientist",
Green Tea Press.

Examples:
--------------------
Calculating the absolute magnitude for a single star
Trying:
>>> apparent_mag = 4.2
>>> apparent_mag_error = 0.3  
>>> distance = 10.0
>>> distance_error = 0.1  
>>> abs_mag, abs_mag_error = absolute_magnitude(apparent_mag, distance, apparent_mag_error, distance_error)
>>> print(f"{abs_mag:.2f} +- {abs_mag_error:.2f}")

Expecting:
>>> 4.20 +- 0.18

Calculating the absolute magnitude for multiple stars
Trying:
>>> apparent_mags = np.array([4.2, 6.5, 5.0, 7.3])
>>> apparent_mag_errors = np.array([0.3, 0.2, 0.4, 0.1])
>>> distances = np.array([10.0, 20.0, 15.0, 25.0])
>>> distance_errors = np.array([0.1, 0.2, 0.15, 0.3])
>>> abs_mags, abs_mags_error = absolute_magnitude(apparent_mags, distances, apparent_mag_errors, distance_errors)
>>> for i in range(len(abs_mags)):
    print(f"Star {i + 1}: Absolute Magnitude = {abs_mags[i]:.2f} ± {abs_mags_error[i]:.2f}")
    
Expecting:
>>> Star 1: Absolute Magnitude = 4.20 ± 0.18
>>> Star 2: Absolute Magnitude = 4.99 ± 0.08
>>> Star 3: Absolute Magnitude = 4.12 ± 0.20
>>> Star 4: Absolute Magnitude = 5.31 ± 0.04

"""

import numpy as np
def absolute_magnitude(apparent_magnitude, distance, apparent_mag_error=0, distance_error=0):
    
    # Calculate the absolute magnitude using the provided formula
    absolute_mag = apparent_magnitude - 2.5 * np.log10((distance/10)**2)

    # Calculate the error in absolute magnitude using error propagation
    if np.any(apparent_mag_error != 0) or np.any(distance_error != 0):
        term1 = 2.5 * (apparent_mag_error / apparent_magnitude)
        term2 = 5 * (distance_error / (distance * np.log(10)))
        absolute_mag_error = np.sqrt(term1**2 + term2**2)
    else:
        absolute_mag_error = 0

    return absolute_mag, absolute_mag_error


# In[ ]:




