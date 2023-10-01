#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''
This function performs sigma rejection on a dataset.

The function takes an array containing the data set 
as an input, a tuple of rejection limits that gives 
the number of standard deviations for each iteration, 
and an optional Boolean mask that has the same shape
as the data and indicates which data points are good.


Parameters
-------------------
data: The input data array.

rejection_limits: A tuple of rejection limits for each iteration.

mask: An optional Boolean mask indicating which data points are good.

Returns
-------------------
mask: Modified mask after sigma rejection is complete.

References
-------------------
. . [1] Downy, A., 2015, "Think Python: How to Think Like a Computer Scientist",
Green Tea Press.


Examples:
--------------------
Trying:
>>> data = np.array([10, 11, 12, 8, 9, 10, 11, 8, 9, 1000])
>>> rejection_limits = (2.0,) 
>>> modified_mask = sigrej(data, rejection_limits, initial_mask)

Expecting:
# A modified mask with 2 sigma rejection


End of docstring. '''
import numpy as np

def sigrej(data, rejection_limits, mask = None):
    
    if mask is None:
        mask = np.ones_like(data, dtype = bool)  # Consider all data points as initially good

    for limit in rejection_limits:
        
        # Calculate mean and standard deviation of the current data points with the mask
        mean = np.mean(data[mask]) # Calculate mean
        std_dev = np.std(data[mask]) # Calculate standard devation

        # Calculate the absolute deviation from the mean
        abs_deviation = np.abs(data - mean)

        # Flag data points as bad (False) if they are beyond the rejection limit
        mask = mask & (abs_deviation <= limit * std_dev)

    return mask


# In[ ]:




