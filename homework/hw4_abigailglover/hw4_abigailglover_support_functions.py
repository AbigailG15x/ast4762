#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This is my square function
# Function's docstring

''' This function explores probability.

The function creates a subsample containing 
N random draws from a given parent sample.
For each sample, the function produces the
ample number, the sample mean, and the sample 
standard deviation in an array with one row 
per sample.

Parameters
-------------------
sample_sizes: This is a list of integers representing
the sample sizes you want to generate samples for
and calculate statistics.

num_repetitions: An integer indicating the number of 
repetitions for each sample size. It specifies how 
many times you want to generate samples of the same 
size.

Returns
-------------------
results_list: Contains batches of results for 
each specified sample size. Each batch is represented
as a NumPy array with three columns: sample number, 
sample mean, and sample standard deviation. The results_list 
contains one batch of results for each sample size specified
in the sample_sizes parameter.

Raises
-------------------
np.random.normal: Could raise exceptions if it encounters 
errors, such as incorrect input arguments or issues related
to the random number generation process. In such cases, these
exceptions would propagate up to the caller of the function, 
but the code itself doesn't contain explicit exception handling.

References
-------------------
. . [1] Downy, A., 2015, "Think Python: How to Think Like a Computer Scientist",
Green Tea Press.

Examples:
--------------------
Trying:
>>> sample_sizes = [1000]
>>> num_repetitions = 10
>>> results = sample_draws(sample_sizes, num_repetitions)
Expecting:
# results is a list containing one batch of results for the sample size 1000.
The batch contains 10 repetitions of samples with sample number, sample mean, 
and sample standard deviation recorded.
    

End of docstring. '''

import numpy as np

def sample_draws(sample_sizes, num_repetitions, batch_size=1000):
    
    # List to store batch results
    results_list = []  
    
    for sample_size in sample_sizes:
        batch_results = np.zeros((num_repetitions, 3))
        
        for i in range(num_repetitions):
            
            # Generate a sample of the specified size from a Gaussian distribution
            sample = np.random.normal(0, 1, sample_size)
            
            # Record sample number, sample mean, and sample standard deviation
            batch_results[i, 0] = i + 1  # Sample number
            batch_results[i, 1] = np.mean(sample)  # Sample mean
            batch_results[i, 2] = np.std(sample)   # Sample standard deviation
        
        results_list.append(batch_results)
    
    return results_list


# In[ ]:




