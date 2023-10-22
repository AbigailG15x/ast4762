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

    # Calculate the error in absolute magnitude
    if np.any(apparent_mag_error != 0) or np.any(distance_error != 0):
        term1 = 2.5 * (apparent_mag_error / apparent_magnitude)
        term2 = 5 * (distance_error / (distance * np.log(10)))
        absolute_mag_error = np.sqrt(term1**2 + term2**2)
    else:
        absolute_mag_error = 0

    return absolute_mag, absolute_mag_error


# In[36]:


"""
This function calculates the absolute magnitude of
a star and its associated error, while also handling 
cases where there may be missing or NaN values in the
input data.

The function Checks if any of the magnitudes or errors 
are missing or NaN and excludes them from the calculation 
of absolute magnitudes. It returns the cleaned-up absolute
magnitudes for the two colors (V and B) and their 
corresponding errors

Parameters
-------------------
apparent_mag1: An array of apparent magnitudes for the V band.

apparent_mag_error1: An array of errors in the apparent magnitudes for the V band.

apparent_mag2: An array of apparent magnitudes for the B band.

apparent_mag_error2: An array of errors in the apparent magnitudes for the B band.

distance: An array of distances to the stars.

distance_error: An array of errors in the distances to the stars.

Returns
-------------------
cleaned_absolute_mag1: An array containing the cleaned up absolute magnitudes for the V band. 
This array excludes any values where either the apparent magnitude or its associated error 
is missing.

cleaned_absolute_mag_error1: An array containing the corresponding errors in the cleaned absolute 
magnitudes for the V band. Like the absolute magnitudes, this array excludes any values with 
missing or NaN data points.

cleaned_absolute_mag2: An array containing the cleaned up absolute magnitudes for the B band. 
This array also excludes values with missing or NaN data.

cleaned_absolute_mag_error2: An array containing the corresponding errors in the cleaned absolute 
magnitudes for the B band. Like the absolute magnitudes, this array excludes values with
missing or NaN data points.

References
-------------------
. . [1] Downy, A., 2015, "Think Python: How to Think Like a Computer Scientist",
Green Tea Press.

Examples:
--------------------

Trying:
>>> apparent_mags1 = np.array([[5.0, 6.8], [9.2, 3.0], [np.nan, np.nan]])
>>> apparent_mag_errors1 = np.array([[0.3, 0.3], [0.4, 0.2], [0.1, 0.1]])
>>> distances1 = np.array([2.1, 10.1, np.nan])
>>> distance_errors1 = np.array([0.1, 0.25, 0.05])

>>> cleaned_abs_mag1, cleaned_abs_mag2, cleaned_abs_mag_error1, cleaned_abs_mag_error2 = clean_absolute_magnitudes(apparent_mags1, apparent_mag_errors1, distances1, distance_errors1)

>>> print("Cleaned Absolute Magnitudes (V):", cleaned_abs_mag1)
>>> print("Cleaned Absolute Magnitude Errors (V):", cleaned_abs_mag_error1)
>>> print("Cleaned Absolute Magnitudes (B):", cleaned_abs_mag2)
>>> print("Cleaned Absolute Magnitude Errors (B):", cleaned_abs_mag_error2)

Expecting:
>>> Cleaned Absolute Magnitudes (V): [8.38890353 9.17839313]
>>> Cleaned Absolute Magnitude Errors (V): [0.18218747 0.12125895]
>>> Cleaned Absolute Magnitudes (B): [10.18890353  2.97839313]
>>> Cleaned Absolute Magnitude Errors (B): [0.15118553 0.17511929]

Trying:

    
Expecting:


"""
import numpy as np

def clean_absolute_magnitudes(apparent_mag1, apparent_mag2, apparent_mag_error1, apparent_mag_error2, distance, distance_error):
    # Check for missing or NaN values and exclude them from the calculation
    valid_indices = ~np.isnan(apparent_mag1) & ~np.isnan(apparent_mag2) & ~np.isnan(apparent_mag_error1) & ~np.isnan(apparent_mag_error2)

    # Clean the input data
    apparent_mag1 = apparent_mag1[valid_indices]
    apparent_mag2 = apparent_mag2[valid_indices]
    apparent_mag_error1 = apparent_mag_error1[valid_indices]
    apparent_mag_error2 = apparent_mag_error2[valid_indices]

    # Calculate the absolute magnitudes
    abs_mag1 = apparent_mag1 - 2.5 * np.log10((distance/10)**2)
    abs_mag2 = apparent_mag2 - 2.5 * np.log10((distance/10)**2)

    # Calculate the error in absolute magnitudes
    term1 = 2.5 * (apparent_mag_error1 / apparent_mag1)
    term2 = 2.5 * (apparent_mag_error2 / apparent_mag2)
    term3 = 5 * (distance_error / (distance * np.log(10)))
    abs_mag_error1 = np.sqrt(term1**2 + term3**2)
    abs_mag_error2 = np.sqrt(term2**2 + term3**2)

    return abs_mag1, abs_mag2, abs_mag_error1, abs_mag_error2


# In[ ]:




