#!/usr/bin/env python
# coding: utf-8

# In[32]:


# This is my square function
# Function's docstring

''' This function takes an integer or an array and provides
the square value in return.

It can accept multiple types of numerical values and
any type of array. You can type the code to create
an array when requested or upload a .txt file containing
the array/number.

Parameters
-------------------
square
    square(input_value): This is the main function of the script. 
    It receives an integer or array input and checks the type of
    value to ensure it is compatible with the function before
    carrying out square parameter.

load_input_from_txt
    load_input_from_txt(filename): Used to load input from a .txt 
    file. It reads the file and, if successful, loads the data.

Returns
-------------------
result: returns square of the value

Other Parameters
-------------------
input: Prompt string that directs the user what to input

Raises
-------------------
ValueError if user does not input supported value/file/array

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
Testing all possible input data types 

    python -m doctest -v doctest_simple_with_docs.py

    # Scalar integer
    Trying:
    >>> square(3)  
    Expecting:
    9

    # Scalar floating-point number
    Trying:
    >>> square(2.5)
    Expecting:
    6.25

    # Negative scalar integer
    Trying:
    >>> square(-4)
    Expecting:
    16

    # Scalar float
    Trying:
    >>> square(0.0)
    Expecting:
    0.0

    # NumPy array
    Trying:
    >>> square(np.array([1, 2, 3]))  
    Expecting:
    array([1, 4, 9])

    # List of float values
    Trying:
    >>> square([1.5, 2.5, 3.5])
    Expecting:
    [2.25, 6.25, 12.25]

    # Loaded input from .txt
    # Homework Problem 2f
    Trying:
    >>> square(load_input_from_txt('test_square_1.txt'))
    Expecting:
    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

    # Loaded input from .txt
    # Homework Problem 2g
    Trying:
    >>> square(load_input_from_txt('test_square_2.txt'))
    Expecting:
    [[  0   1   4   9  16]
     [ 25  36  49  64  81]
     [100 121 144 169 196]
     [225 256 289 324 361]
     [400 441 484 529 576]]
    

End of docstring. '''

# Main function
# Import libraries for many data types
import numpy as np
import numbers
import ast

# Define the function
def square(input_value):


    # Determine if the input value is a scalar
    if isinstance(input_value, numbers.Number):

        # Square the scalar
        return input_value ** 2

    # Determine if the input value is an array (NumPy or list)
    elif isinstance(input_value, (np.ndarray, list)):

        # Convert lists to NumPy arrays if needed
        if isinstance(input_value, list):
            input_value = np.array(input_value)

        # Square the array
        return np.square(input_value)

    # If it's neither a scalar nor an array, raise an error
    else:
        raise ValueError("Input must be a scalar or an array.")

# Function to load input from a .txt file
def load_input_from_txt(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
        return ast.literal_eval(content)  # Safely evaluate Python literals
    except Exception as e:
        raise ValueError("Error loading input from the .txt file: " + str(e))


# In[ ]:





# In[ ]:




