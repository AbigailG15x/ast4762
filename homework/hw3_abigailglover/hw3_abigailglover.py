#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Abigail Glover
# Homework 3
# 9/7/2023

##############################

# Import libraries (all pulled from template)
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt 
from astropy.io import fits
import os
import hw3_abigailglover_support_functions 

# Problem 2 a - e completed in function and 
# documented in log

print('Problem 2')
print('f.)\n')

# Array test square
test_square_1 = np.arange(10)

# Verify array
print('Input array:')
print(test_square_1)

# Call function to square
result = hw3_abigailglover_support_functions.square(test_square_1)

# Print the result
print('\nOutput array:')
print(result)


# In[3]:


# Next problem
print('\n\nProblem 2')
print('g.)\n')

# Create new array
test_square_2 = np.arange(25, dtype=float).reshape(5, 5)

# Print new array
print('New input array:')
print(test_square_2)

# Call function to square
result2 = hw3_abigailglover_support_functions.square(test_square_2)

# Print new output
print('\nNew output array:')
print(result2)


# In[6]:


# Print problem information
# NOTE: I was not able to get this sections to run correctly
print('Problem 3')

# Define the low and high values
low = 0
high = 10

# List numbers from assignment
numbers_to_plot = [1, 2.5, 4, 5.5, 7]

# Define the number of points
num_points = 100

# Define the filename to save the plot
saveplot = "hw3_abigailglover_problem3_graph1.pdf" 

# Call the function
hw3_abigailglover_support_functions.squareplot(numbers_to_plot, num_points, saveplot)


# In[9]:


# Problem 4
print('Problem 4\n')
print('Plot the file m42_40min_ir.fits\n')

# Get a FITS file and show info about the file
# From week 3 lecture
fits.info('m42_40min_ir.fits') 

# Read image data from file
# From week 3 lecture
im = fits.getdata('m42_40min_ir.fits') 

# Create a grayscale plot with a flipped y-axis
plt.imshow(np.flipud(im), cmap='gray', extent=[0, im.shape[1], 0, im.shape[0]])

# Annotate the image with the object's name and your name as a title

plt.title("M42 Nebula - Abigail Glover")

# Add appropriate axis labels
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

# Save the final plot as a PNG file
plt.savefig('hw3_abigailglover_problem4_graph1.png', format='png')

# Display the plot (optional)
plt.show()

# Answering the question in the hw file
print('What object is it?')
print('The plot shows Messier 42, also known as the Orion Nebula.')


# In[19]:


# Problem 5
# Read image data from file
im = fits.getdata('m42_40min_ir.fits')

# Call the linear scaling function
scaled_im = scale(im)

# Calculate the difference between the original and scaled datasets
difference = im - scaled_im

# Create a figure with subplots to display the original, scaled, and difference images
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Display the original image
axs[0].imshow(im, cmap='gray', extent=[0, im.shape[1], 0, im.shape[0]])
axs[0].set_title("Original Image")
axs[0].axis('off')

# Display the scaled image
axs[1].imshow(scaled_im, cmap='gray', extent=[0, scaled_im.shape[1], 0, scaled_im.shape[0]])
axs[1].set_title("Scaled Image")
axs[1].axis('off')

# Display the difference image with a colorbar
cax = axs[2].imshow(difference, cmap='coolwarm', extent=[0, difference.shape[1], 0, difference.shape[0]])
axs[2].set_title("Difference Image")
axs[2].axis('off')

# Add a colorbar for the difference image
fig.colorbar(cax, ax=axs[2], orientation='vertical', label='Difference')

# Save the final plot as a PNG file
plt.savefig('hw3_abigailglover_problem4_graph2.png', format='png')

# Show the plot
plt.show()


# In[ ]:




