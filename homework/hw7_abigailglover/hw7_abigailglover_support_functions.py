#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries
import numpy as np

def normmedcomb(data, norm_region=None):
    
    '''
    This function implements the normal median combination method.
    
    This function performs normalized median combination on a 3D 
    array of dark-subtracted data containing sky flux. A tuple 
    containing the corner coordinates of the normalization region
    is used as an optional keyword with the default being the 
    entire array. This function then returns a tuple containing
    the normalized, median-combined, 2D array, and a 1D array 
    containing the normalization factors

    Parameters
    ----------
    data: a 3D numpy array containing dark-subtracted data, 
    representing the sky flux. Each dimension has a specific 
    meaning:
        - The first dimension represents the different frames or exposures.
        - The second and third dimensions represent the spatial dimensions, 
        typically the y and x coordinates.
        
    norm_region: Optional keyword that defines the corner coordinates of 
    the normalization region within the data array in the format
      
    Returns
    -------
    median_combined: Stores the result of performing median combination
    on the normalized frames. This variable represents the median-combined 2D 
    array of the input data after normalization.
    
    normalization_factors: A 1D NumPy array that stores the normalization 
    factors for each frame in the input data.

    '''
    
    # Make a copy of the input data
    data_copy = data.copy()

    # Define the default normalization region as the entire array
    if norm_region is None:
        norm_region = ((0, 0), data.shape[2:])

    # Extract the coordinates of the normalization region
    (y1, x1), (y2, x2) = norm_region

    # Create a different array to store the normalization factors for each frame
    normalization_factors = np.empty(data.shape[0])

    # Iterate through each frame in the data
    for i in range(data.shape[0]):
        # Extract the current frame
        frame = data_copy[i, y1:y2, x1:x2]

        # Calculate the median of the current frame
        median_value = np.median(frame)

        # Normalize the current frame by dividing by the median
        data_copy[i] /= median_value

        # Store the normalization factor for this frame
        normalization_factors[i] = median_value

    # Compute the median combination of the normalized frames
    median_combined = np.median(data_copy, axis=0)

    return median_combined, normalization_factors

