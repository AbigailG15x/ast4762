#!/usr/bin/env python
# coding: utf-8

# In[20]:


# Abigail Glover
# HW 4
# 9/13/2023

###############################

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm # Gaussian normal distribution

# Problem 2 print statement
print('Problem 2')
print('Part a.)')

# Specify parameters from homework file
sigma = 55
cx = 13
N = 10000

# From NumPy User Guide 
sample = np.random.normal(loc = sigma, scale = cx, size = N)


# In[21]:


# Print problem number
# Problem 2 (continued)
print('Part b.)')

# Plot the histogram with specified bins and width
plt.hist(sample, bins = np.arange(0, 101, 1), density = True, label = 'Generated Samples', edgecolor = 'black', color = 'firebrick')

# Label the axes
plt.xlabel('Value', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)
plt.title('Random Sampling of Gaussian Distribution', fontsize = 16)

# Create a legend
legend = plt.legend(fontsize = 'medium')

# Found a great way to make my legend nicer
# Following code modified from Stackflow
frame = legend.get_frame() #sets up for color, edge, and transparency
frame.set_facecolor('grey') #color of legend
frame.set_edgecolor('black') #edge color of legend
frame.set_alpha(0.2) #deals with transparency

# Save the plot as a png
plt.savefig('hw4_abigailglover_problem2_graph1.png')

# Show plot
plt.show()


# In[22]:


# Print problem number
# Problem 2 (continued)
print('Part c.)')

# Define bins
bin_edges = np.arange(0, 101, 1)

# Calculate the center of each bin
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Calculate the expected number of draws per bin
expected_counts = N * norm.pdf(bin_centers, loc = sigma, scale = cx)

# Plot the Histogram
plt.hist(sample, bins = np.arange(0, 101, 1), density = True, edgecolor = 'black', color = 'firebrick')

# Plot the Gaussian and ensure that it is scaled to the histogram
# Base code from GeeksforGeeks and modified for my parameters
plt.plot(bin_centers, norm.pdf(bin_centers, sigma, cx), label = 'Expected Gaussian', color = 'gold', linewidth = '3')

# Label axes
plt.xlabel('Value', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)
plt.title('Random Sampling of Gaussian Distribution with Expected Gaussian', fontsize = 14)

# Create a legend
plt.legend(title = 'Legend', fontsize = 'small')

# Save the plot as a PNG file
plt.savefig('hw4_abigailglover_problem2_graph2.png')

# Show the plot
plt.show()


# In[23]:


# Print problem number
print('Problem 3')
print('Parts a - c.)\n')

# Call function
import hw4_abigailglover_support_functions

# Sample sizes and number of repetitions
sample_sizes = [10, 100, 1000, 10000, 100000, 1000000]
num_repetitions = 10

# Lists to store sample sizes and standard deviations of means
sample_sizes_log = []  
std_dev_means_log = []  

# Create or open a text file to store the results
with open("hw4_abigailglover_problem3_data.txt", "a") as file:
    for sample_size in sample_sizes:
        
        # List to store results for each sample size
        all_results = []

        for _ in range(num_repetitions):
            
            # Generate and record samples
            results_list = hw4_abigailglover_support_functions.sample_draws([sample_size], num_repetitions)

            # Update the results to the list
            all_results.append(results_list[0]) 

        # Convert the list of lists to a NumPy array
        all_results = np.array(all_results)

        # Add a comment stating the number of draws
        file.write("# Number of Draws: {}\n".format(sample_size))

        # Save all results to the text file
        for results in all_results:
            np.savetxt(file, results, fmt='%d %.6f %.6f', delimiter='\t', header='Sample Number\tSample Mean\tSample Std Dev')

        # Calculate standard deviation of the means for all repetitions
        std_dev_means = np.std(all_results[:, :, 1])  

        # Append sample size and standard deviation of the means to lists
        sample_sizes_log.append(sample_size)
        std_dev_means_log.append(std_dev_means)

        # Print sample size and standard deviation of the means
        print("Sample Size: {}, Standard Deviation of Means: {:.6f}".format(sample_size, std_dev_means))
        
        # Append a comment with results to the same text file
        file.write("# Sample Size: {}, Standard Deviation of Means: {:.6f}\n".format(sample_size, std_dev_means))


# In[24]:


# Print problem number
print('Part d.)')

# Create a log-log plot
plt.figure(figsize=(8, 6)) # Plot the figure
plt.loglog(sample_sizes_log, std_dev_means_log, marker='o', linestyle='-', color='firebrick') # Plot the log log
plt.xlabel('(Log) Sample Size') # Label the x-axis
plt.ylabel('(Log) Standard Deviation of Means') # Label the y-axis
plt.title('Standard Deviation of the Mean vs. Sample Size') # Label the plot
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('hw4_abigailglover_problem3_graph1.png')

# Show the plot
plt.show()


# In[26]:


# Print problem number
print('Problem 4')
print('Work for this problem can be found in the file\
 labeled hw4_abigailglover_problem4_data.pdf')


# In[ ]:




