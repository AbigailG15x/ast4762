#!/usr/bin/env python
# coding: utf-8

# In[37]:


# Problem 3

# Square Plot
# Function's docstring


''' This function plots the squares of numbers.

This function has three positional arguments: the 
low end of the range, the high end of the range 
(inclusive), and the number of points to plot over 
this range.

Parameters
-------------------
low: Demonstrates the low end of the range.
high: Demonstrates the high end of the range (inclusive).
num_points: The number of points to plot over the range.

Returns
-------------------
saveplot: Saves the plot as a file

Other Parameters
-------------------

Raises
-------------------


Notes
-------------------

References
-------------------
. . [1] Downy, A., 2015, "Think Python: How to Think Like a Computer Scientist",
Green Tea Press, pgs(17 - 26, 41-45).

Other Sources:
- https://pymotw.com/2/doctest/

Examples:
--------------------

End of docstring. '''

# Import libraries
import numpy as np
import numbers
import ast
import hw3_abigailglover_support_functions

def squareplot(nums, num_points, saveplot):

# Create an array for each number
    x = {}
    y = {}

    for num in numbers_to_plot:
        x[num] = np.linspace(num, num,  num_points)
        y[num] = square(x[num])

    for num in numbers_to_plot:
        plt.plot(x[num], y[num], label = f"Number {num}")

    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("Square Function")

    plt.legend()

    plt.show()


    # Save the plot as an image file
    if saveplot is not None:
        if saveplot is True:
            # Default filename if saveplot is True
            saveplot = "hw3_abigailglover_problem3_graph1.pdf"
        plt.savefig(saveplot, format="pdf")




# In[ ]:




