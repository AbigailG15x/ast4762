#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from astropy.io import fits
import numpy as np

# Part a
def read_and_extract_data(folder_path, start_set, num_sets):

    '''
    Read FITS files from a specified folder, extract header information, and organize data into arrays.

    Parameters
    ----------
    folder_path : (str) The path to the folder containing FITS files.

    start_set : (int) The starting subarray set number.

    num_sets : (int) The number of subarray sets to process.

    Returns
    -------
    data_cube : (numpy.ndarray) A 3D array containing the data from all specified subarray sets.

    frame_params_array : (numpy.ndarray) A 2D array containing information about each frame, 
    including frame number and observation time.

    Raises
    ------
    ValueError if the specified folder_path does not exist or if there are no FITS files in the folder.

    Notes
    -----
    This routine performs the following tasks:
    (Part a) Reads the first data block (BCD) to extract information such as image cube size and prints 
             the number of frames in a given subarray set.
    (Part b) Allocates a 3D data cube and a 2D frame-parameter array, printing the shape of each array.
    (Part c) Loops over the specified number of subarray sets, reading data and headers, populating the 
             data cube, and extracting relevant header information.
        - (Part c i.) Reads the data and header.
        - (Part c ii.) Puts the data into the 3D data cube in blocks of 64 frames.
        - (Part c iii.) Finds the observation time in the header, puts it in the frame parameters array 
                        along with the frame number, and calculates the timing of subsequent frames in the set.
                        
    '''
    
    # Initialize arrays
    data_cube = None
    frame_params_list = []
    
    # List all files in the folder
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".fits")])
    
    for i, file_name in enumerate(file_list[start_set * 100: (start_set + num_sets) * 100]):
        file_path = os.path.join(folder_path, file_name)
        
        # (Part c i.) 
        # Read the data and header.
        with fits.open(file_path) as hdul:
            header = hdul[0].header
            data = hdul[0].data
            
            # (Part b) 
            # Allocate a 3D data cube
            if data_cube is None:
                data_cube = np.empty((num_sets * 64, *data.shape))
            
            # (Part c ii.) 
            # Put data into the 3D data cube in blocks of 64 frames
            data_cube[i * 64: (i + 1) * 64, :, :] = data
            
            # Extract header information
            obs_time = header["DATE-OBS"]  
            frame_number = i + 1
            
            # (Part c iii.) 
            # Find the observation time in the header and put it in the frame parameters with the frame number
            frame_interval = header["FRAMTIME"]
            mid_time = obs_time + 0.5 * header["EXPTIME"]
            frame_times = mid_time + np.arange(0, data.shape[0]) * frame_interval
            
            # Populate frame parameters list with information
            frame_params = {
                "frame_number": frame_number,
                "obs_time": obs_time,
                "frame_times": frame_times,
                
            }
            frame_params_list.append(frame_params)
            
            # Print filename and DATE-OBS for every 10th file
            if i % 10 == 0:
                print("Filename:", file_name)
                print("DATE-OBS:", header["DATE-OBS"])
                print()
    
    if data_cube is not None:
        # Convert frame_params_list to a structured numpy array
        frame_params_array = np.array(
            [(params["frame_number"], params["obs_time"]) for params in frame_params_list],
            dtype=[("frame_number", int), ("obs_time", object)])  # Adjust the dtype as needed
        
        # Print the shape of arrays
        print("Shape of 3D data cube:", data_cube.shape)
        print("Shape of frame parameters array:", frame_params_array.shape)
    else:
        print("No FITS files found or folder_path does not exist.")
    
    return data_cube, frame_params_array

