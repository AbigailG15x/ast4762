#!/usr/bin/env python
# coding: utf-8

# In[358]:


# Abigail Glover
# AST 4762 - Midterm
# Oct 12, 2023

##############################


# In[430]:


# Import libraries
import pandas as pd 
import midterm_abigailglover_support_functions
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits 


# In[361]:


# Print problem number
print('Problem 2\n')

# Part a
print('a.) What is the definition of an open and a globular star cluster?')

# Answer
print('\n- An open star cluster is a cluster consisting of a few tens to a few\
\nhundred stars that are loosely bound together, and is not very stable.')

print('\n- A globular star cluster is a cluster consisting of tens of thousands\
\nto millions of stars that are tightly gravitationally bound, and is very stable.')

# Sources for part a
    # ESA Hubble: https://esahubble.org/wordbank/open-cluster/
    # ESA Hubble: https://esahubble.org/wordbank/globular-cluster/
    # Oxford Dictionary: https://www.oed.com/search/dictionary/?scope=Entries&q=globular+cluster
    # Oxford Dictionary: https://www.oed.com/search/dictionary/?scope=Entries&q=open+cluster

# Part b
print('\n\nb.) Name two differences between an open and a globular star cluster.')

# Answer
print('\n- Open star clusters are typically found in irregular and spiral galaxies\
\nwhereas globular star clusters are associated with all types of galaxies.')

print('\n- Open star clusters exhibit a range of both young and old star ages\
\nwhereas globular clusters are populated by older, redder stars.')

# Sources for part b
    # ESA Hubble: https://esahubble.org/wordbank/open-cluster/
    # ESA Hubble: https://esahubble.org/wordbank/globular-cluster/

# Part c
print('\n\nc.) What is the difference between the apparent and absolute magnitude of a star?')

# Answer
print('\nApparent magnitude refers to the brightness of a star as seen from Earth, whereas absolute\n\
magnitude refers to the brightness of a star from a standard distance of 10 parsecs.')

# Sources for part c
    # Perkins School for the Blind: https://www.perkins.org/resource/apparent-vs-absolute-magnitude-stars-interactive-model/#:~:text=absolute%20magnitude%20%E2%80%93%20a%20measure%20of,star%20as%20seen%20from%20Earth
    # Learn the Sky: https://www.learnthesky.com/blog/apparent-versus-absolute-magnitude

# Part d
print('\n\nd.) What is a color magnitude diagram (CMD) and why is it important for star clusters?')

# Answer
print('\nA color magnitude diagram (CMD) shows the relationship between the absolute magnitude of stars\n\
and their "colors" (often referring to their surface temperature, chemical composition, and mass). It allows\n\
researchers the ability to better understand the physical properties of stellar populations at any age. \n\
It is a derivation of the Hertzsprung-Russell diagram, but specifically observes star clusters rather than\n\
individual stars.')

# Sources for part d
    # Astronomy Online: http://astronomyonline.org/Astrophotography/CMDDiagram.asp
    # Britannica: https://www.britannica.com/science/spectrum
    # Springer: https://link.springer.com/content/pdf/10.1007/978-3-642-27851-8_186-1.pdf

# Part e
print('\n\ne.) How does the CMD evolve over time?')

# Answer
print('\nOver time, the CMD can demonstrate the various evolutionary stages of a specific population. From\n\
turnoff points from the Main Belt Sequence to mass exchange in binary systems. The CMD can offer an overview\n\
of the history of a cluster and provide more details about where it is heading.')

# Sources
    # Astronomy Online: http://astronomyonline.org/Astrophotography/CMDDiagram.asp
    # Springer: https://link.springer.com/content/pdf/10.1007/978-3-642-27851-8_186-1.pdf

# Part f
print('\n\nf.) Name three things we need to take into account in the processes of getting a CMD')
print('and fitting it with isochrones.')

# Answer
print('\n1.) The age range of the population.')
print('2.) Choosing an isochrone that best represents the metallicity and age of the population.')
print('3.) The quality of the data being used.')

# Sources
    # Fitting isochrones to open cluster photometric data: https://www.aanda.org/articles/aa/pdf/2010/08/aa13677-09.pdf
    # Stellar Isochrone: https://en.wikipedia.org/wiki/Stellar_isochrone
    # Stellar models and isochrones from low-mass to massive stars: https://www.aanda.org/articles/aa/full_html/2019/04/aa35051-19/aa35051-19.html


# In[362]:


# Print problem number
print('\n\nProblem 3')

# Print question from midterm
print('\nWhat do you notice about the dataset for "Star Clusters in M33"? Name a few reasons why this might happen.')

# Answer
print('\nWhen observing the dataset I noticed missing values for some of the B-V and U-B values and their associated errors.')
print('In total, there are about 12 rows missing these four values, but the magnitude for V is still present for each.')
print('There is no case where either B-V OR U-B is missing, it is both of them for every case, meaning that the data for B')
print('is the missing value, as both of these data points rely on B for their final values. This could be due to:')
print('\n- Observational limitations due to weather, visibility, instrument downtime, etc')
print('- Data processing issues, such as failed measurements.')
print('- Objects being outside the sensitivity range of the instrument.')
print('- and more')
print('\nIt is important to note that similar V-band ranges had values for B-V and U-B, meaning that this was not a predictable')
print('error at any specific V-band magnitude or range of V-band values.')


# In[363]:


# Print Problem statement
print('\n\nProblem 4')

# Import libraries 
import numpy as np 
import matplotlib.pyplot as plt

# Let user know what has been done
print('\nLibraries have been imported')


# In[364]:


# Print problem statement
print('\n\nProblem 5')

# Print problem
print('\na) A star with a V magnitude of 10 ± 0.1 mag at a distance of 3.1 ± 0.2 pc.')

# List given data
apparent_mag = 10.0
apparent_mag_error = 0.1
distance = 3.1
distance_error = 0.2

# Call to function
abs_mag, abs_mag_error = midterm_abigailglover_support_functions.absolute_magnitude(apparent_mag, distance, apparent_mag_error, distance_error)

# Print results
print(f"\nAbsolute Magnitude: {abs_mag:.2f} ± {abs_mag_error:.2f}")

# Print problem
print('\n\nb) A group of stars located at 2.1 ± 0.1 pc from us and with V magnitudes of\n\
5.0 ± 0.3, 6.8 ± 0.3 and 9.2 ± 0.4 mag.\n')

# List given data
apparent_mags = np.array([5.0, 6.8, 9.2])
apparent_mag_errors = np.array([0.3, 0.3, 0.4])
distances = np.array([2.1, 2.1, 2.1])
distance_errors = np.array([0.1, 0.1, 0.1])

# Initialize empty arrays to store the absolute magnitudes and errors
abs_mags = np.zeros_like(apparent_mags)
abs_mag_errors = np.zeros_like(apparent_mags)

# Calculate the absolute magnitudes for each star
for i in range(len(apparent_mags)):
    # Call the function
    abs_mags[i], abs_mag_errors[i] = midterm_abigailglover_support_functions.absolute_magnitude(apparent_mags[i], distances[i], apparent_mag_errors[i], distance_errors[i])

# Print the results
for i in range(len(apparent_mags)):
    print(f"Star {i + 1}:")
    print(f"Absolute Magnitude: {abs_mags[i]:.2f} ± {abs_mag_errors[i]:.2f}")
    print()

# Print problem
print('\nc) a group of stars with V magnitudes of 3.1 ± 0.3, 3.8 ± 0.3 and 3.2 ± 0.4\n\
mag at a distance of 2.1 ± 0.1 pc, 5.10 ± 0.14 pc and 10.10 ± 0.25 pc respectively.\n')

# List given data
apparent_mags = np.array([3.1, 3.8, 3.2])
apparent_mag_errors = np.array([0.3, 0.3, 0.4])
distances = np.array([2.1, 5.10, 10.10])
distance_errors = np.array([0.1, 0.14, 0.25])

# Initialize empty arrays to store the absolute magnitudes and errors
abs_mags = np.zeros_like(apparent_mags)
abs_mag_errors = np.zeros_like(apparent_mags)

# Calculate the absolute magnitudes for each star
for i in range(len(apparent_mags)):
    # Call the function
    abs_mags[i], abs_mag_errors[i] = midterm_abigailglover_support_functions.absolute_magnitude(apparent_mags[i], distances[i], apparent_mag_errors[i], distance_errors[i])

# Print the results
for i in range(len(apparent_mags)):
    print(f"Star {i + 1}:")
    print(f"Absolute Magnitude: {abs_mags[i]:.2f} ± {abs_mag_errors[i]:.2f}")
    print()


# In[365]:


# Print problem statement
print('\n\nProblem 6\n')

# Given data
# V magnitudes
apparent_mag1 = np.array([6.2, np.nan, 7.2, 6.0])
# B magnitudes
apparent_mag2 = np.array([5.1, 5.1, 3.2, 3.1])
# V Error
apparent_mag_error1 = np.array([0.4, np.nan, 0.4, 0.3])
# B Error
apparent_mag_error2 = np.array([0.1, 0.2, 0.2, 0.1])

# Distance data
distance = 10.0
distance_error = 0.1

cleaned_abs_mag1, cleaned_abs_mag2, cleaned_abs_mag_error1, cleaned_abs_mag_error2 = midterm_abigailglover_support_functions.clean_absolute_magnitudes(apparent_mag1, apparent_mag2, apparent_mag_error1, apparent_mag_error2, distance, distance_error)

# Calculate B-V magnitudes
B_V_magnitudes = cleaned_abs_mag2 - cleaned_abs_mag1

# Calculate B-V magnitudes errors
B_V_mag_err = apparent_mag_error2 - apparent_mag_error1

# Print the results for V vs B-V
print("V vs B-V CMD:\n")
print("V Absolute Magnitudes:", cleaned_abs_mag1)
print("B-V Absolute Magnitudes:", B_V_magnitudes)
print("Absolute Magnitude Errors (V):", cleaned_abs_mag_error1)
print("Absolute Magnitude Errors (B-V):", cleaned_abs_mag_error2)

# Given data
# B magnitudes
apparent_mag1 = np.array([5.1, 5.1, 3.2, 3.1])
# U magnitudes
apparent_mag2 = np.array([8.1, 8.9, np.nan, 10.5])
# B Error
apparent_mag_error1 = np.array([0.1, 0.2, 0.2, 0.1])
# U Error
apparent_mag_error2 = np.array([0.1, 0.1, np.nan, 0.2])

# Distnce Data
distance = 10.0
distance_error = 0.1

# Calculate U-B magnitudes
U_B_magnitudes = cleaned_abs_mag2 - cleaned_abs_mag1

# Calculate U-B magnitudes errors
U_B_mag_err = apparent_mag_error2 - apparent_mag_error1

# Print the results for B vs U-B CMD
print('\n')
print("B vs U-B CMD:\n")
print("B Absolute Magnitudes:", cleaned_abs_mag1)
print("U-B Absolute Magnitudes:", cleaned_abs_mag2)
print("Absolute Magnitude Errors (B):", cleaned_abs_mag_error1)
print("Absolute Magnitude Errors (U-B):", cleaned_abs_mag_error2)


# In[366]:


# Print problem statement
print('\n\nProblem 7')
print('Part a.)')

# Specify the delimiter to be a comma
delimiter = ','

# Skip the first row 
skiprows = 1

# Read the data from the file and set column names
column_names = ["Starnum", "u[mag]", "uerr[mag]", "b[mag]", "berr[mag]", "v[mag]", "verr[mag]", "i[mag]", "ierr[mag]"]
omega_cen_data = pd.read_csv("omega_cen_observations.dat", delimiter=delimiter, skiprows=skiprows, names=column_names)

# Print sample data
#print(omega_cen_data.head())

# Let the user know what's been done
print('\nData has been read in')


# In[367]:


# Print problem statement
print('\nPart b.)')

# Check for missing data and count the missing values
missing_data = omega_cen_data.isnull().sum()
total_missing = missing_data.sum()

print(f'\nTotal Missing Data Points: {total_missing}')


# In[368]:


# Print problem statement
print('\nPart c.)')

# Set variables based on reading
omega_cen_dist = 5.5
dist_err = 0.2

# Print values for user
print(f"\nThe distance from omega Centauri to Earth is: {omega_cen_dist:.2f} ± {dist_err:.2f}")


# In[369]:


#print(omega_cen_data.keys())
#print(omega_cen_data.head())  # Print the first few rows of data


# In[439]:


print('\nPart d.)-e.)')

# First set of data (B and U bands)
abs_mag_B1, abs_mag_U1, abs_mag_error_B1, abs_mag_error_U1 = midterm_abigailglover_support_functions.clean_absolute_magnitudes(
    omega_cen_data["b[mag]"], omega_cen_data["u[mag]"], omega_cen_data["berr[mag]"], omega_cen_data["uerr[mag]"],
    omega_cen_dist, dist_err)

# Second set of data (B and V bands)
abs_mag_B2, abs_mag_V2, abs_mag_error_B2, abs_mag_error_V2 = midterm_abigailglover_support_functions.clean_absolute_magnitudes(
    omega_cen_data["b[mag]"], omega_cen_data["v[mag]"], omega_cen_data["berr[mag]"], omega_cen_data["verr[mag]"],
    omega_cen_dist, dist_err)

# Third set of data (V and I bands)
abs_mag_V1, abs_mag_I1, abs_mag_error_V1, abs_mag_error_I1 = midterm_abigailglover_support_functions.clean_absolute_magnitudes(
    omega_cen_data["v[mag]"], omega_cen_data["i[mag]"], omega_cen_data["verr[mag]"], omega_cen_data["ierr[mag]"],
    omega_cen_dist, dist_err)

# Set plot size and centering for titles
plt.figure(figsize=(12, 6))

# U vs B-U CMD for the first set of data
plt.subplot(131)
plt.errorbar(abs_mag_B1 - abs_mag_U1, abs_mag_U1, xerr=np.sqrt(abs_mag_error_B1 ** 2 + abs_mag_error_U1 ** 2), yerr=abs_mag_error_U1, fmt='o', markersize=4)
plt.title("Absolute Magnitude for Stars in \nOmega Centauri: U vs B-U", multialignment='center')
plt.xlabel("B-U")
plt.ylabel("U")
plt.ylim(38,28)

# Save the plot as PDF
plt.savefig("midterm_abigailglover_problem7_graph1.pdf") 

# V vs B-V CMD for the second set of data
plt.subplot(132)
plt.errorbar(abs_mag_B2 - abs_mag_V2, abs_mag_V2, xerr=np.sqrt(abs_mag_error_B2 ** 2 + abs_mag_error_V2 ** 2), yerr=abs_mag_error_V2, fmt='o', markersize=4)
plt.title("Absolute Magnitude for Stars in \nOmega Centauri: V vs B-V", multialignment='center')
plt.xlabel("B-V")
plt.ylabel("V")
plt.ylim(38,28)

# Save the plot as PDF
plt.savefig("midterm_abigailglover_problem7_graph2.pdf") 

# I vs V-I CMD for the third set of data
plt.subplot(133)
plt.errorbar(abs_mag_V1 - abs_mag_I1, abs_mag_I1, xerr=np.sqrt(abs_mag_error_V1 ** 2 + abs_mag_error_I1 ** 2), yerr=abs_mag_error_I1, fmt='o', markersize=4)
plt.title("Absolute Magnitude for Stars in \nOmega Centauri: V vs V-I", multialignment='center')
plt.xlabel("V-I")
plt.ylabel("I")
plt.ylim(38,28)

# Save the plot as PDF
plt.savefig("midterm_abigailglover_problem7_graph3.pdf")

# Apply tight_layout() to improve spacing and alignment
plt.tight_layout()

# Show the plots
plt.show()


# In[371]:


# Print problem statement
print('\n\nProblem 8')
print('Part a.)')

# Specify the file path
file_path = 'open_clusters_compilation.tsv'

# Specify the delimiter to be a semicolon
delimiter = ';'

# Define the values that should be treated as NaN
na_values = ['      ']

# Skip the first two rows
skiprows = 2

# Read the data from the file and set column names
column_names = ["_RAJ2000", "_DEJ2000", "RAJ2000", "DEJ2000", "NGC", "Umag", "e_Umag", "Bmag", "e_Bmag", "Vmag", "e_Vmag", "Rcmag", "e_Rcmag", "Icmag", "e_Icmag"]
open_clust_data = pd.read_csv(file_path, delimiter=delimiter, skiprows=skiprows, names=column_names, na_values=na_values)

# Let user know what was done
print('\nData has been read in')


# In[372]:


# Print problem statement
print('Part b.)')

# Describe the data
non_numeric_description = open_clust_data.describe(include='all') # Use non-numeric description to include all values

# Print the results
print(non_numeric_description)


# In[373]:


# Print problem statement
print('Part c.)\n')

# Define NGC values for each cluster
ngc_values_cluster1 = [2232]
ngc_values_cluster2 = [2516]
ngc_values_cluster3 = [2547]
ngc_values_cluster4 = [4755]

# Filter the data for each cluster based on NGC values
cluster1_data = open_clust_data[open_clust_data['NGC'].isin(ngc_values_cluster1)]
cluster2_data = open_clust_data[open_clust_data['NGC'].isin(ngc_values_cluster2)]
cluster3_data = open_clust_data[open_clust_data['NGC'].isin(ngc_values_cluster3)]
cluster4_data = open_clust_data[open_clust_data['NGC'].isin(ngc_values_cluster4)]

# Let user know what was done
print('\nData was organized by respective cluster.')


# In[384]:


import matplotlib.pyplot as plt

# Print problem statement
print('Part d.)\n')

# Create a 2x2 figure for the CMDs
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Function to plot CMD for a specific cluster
def plot_cmd_for_cluster(ax, cluster_data, title):
    
    # Convert 'Bmag' and 'Vmag' columns to numeric, setting non-numeric values to NaN
    cluster_data.loc[:, 'Bmag'] = pd.to_numeric(cluster_data['Bmag'], errors='coerce')
    cluster_data.loc[:, 'Vmag'] = pd.to_numeric(cluster_data['Vmag'], errors='coerce')
    
    # Filter out rows with missing or non-numeric values
    valid_rows = np.isfinite(cluster_data['Bmag']) & np.isfinite(cluster_data['Vmag'])
    filtered_data = cluster_data.loc[valid_rows]

    # Plot the CMD and invert the y-axis
    ax.scatter(filtered_data['Bmag'] - filtered_data['Vmag'], filtered_data['Vmag'], s=5)
    ax.set_xlabel('B-V')
    ax.set_ylabel('V')
    ax.set_title(title)
    
    # Invert the y-axis
    ax.invert_yaxis()

# Plot CMDs for all 4 clusters
plot_cmd_for_cluster(axes[0, 0], cluster1_data, 'NGC 2232 CMD')
plot_cmd_for_cluster(axes[0, 1], cluster2_data, 'NGC 2516 CMD')
plot_cmd_for_cluster(axes[1, 0], cluster3_data, 'NGC 2547 CMD')
plot_cmd_for_cluster(axes[1, 1], cluster4_data, 'NGC 4755 CMD')

# Adjust layout and save the figure as a PDF
plt.tight_layout()
plt.savefig('midterm_abigailglover_problem8_graph1.pdf')

# Show the plots
plt.show()


# In[419]:


# Print problem statement
print('\n\nProblem 9')
print('Part a.)')

# let user know what was done
print('\nDataset was saved as young_star_isochrones.txt')


# In[420]:


# Print problem statement
print('\nPart b.)')

# Specify the file path
file_path = 'young_star_isochrones.txt'

# Set the delimiter
delimiter = r'\s+'

# Read the data without specifying the header
isochrones_data = pd.read_csv(file_path, delimiter=delimiter, engine='python', comment='#', header=None)

# Manually specify column names
column_names = [
    "Zini", "MH", "logAge", "Mini", "int_IMF", "Mass", "logL", "logTe",
    "logg", "label", "McoreTP", "C_O", "period0", "period1", "period2", "period3",
    "period4", "pmode", "Mloss", "tau1m", "X", "Y", "Xc", "Xn", "Xo", "Cexcess", "Z",
    "mbolmag", "Umag", "Bmag", "Vmag", "Rmag", "Imag", "Jmag", "Hmag", "Kmag"
]

# Assign the column names to the DataFrame
isochrones_data.columns = column_names

# Let the user know what has been done
print('\nData has been read and organized.')


# In[421]:


# Print problem statement
print('\nPart c.)-d.)\n')

# Remove # symbol from column names
isochrones_data.columns = isochrones_data.columns.str.strip('# ')

# Get ages and metallicities
ages = isochrones_data['logAge'].unique()
metallicities = isochrones_data['Zini'].unique()

# Extract U, B, and V magnitudes (assuming they are absolute magnitudes)
u_magnitudes = isochrones_data['Umag']
b_magnitudes = isochrones_data['Bmag']
v_magnitudes = isochrones_data['Vmag']

# Print the extracted information
print("Unique Ages:", ages)
print("Unique Metallicities:", metallicities)
print(f"\n\nU Magnitudes (Absolute):\n{u_magnitudes}")
print(f"\n\nB Magnitudes (Absolute):\n{b_magnitudes}")
print(f"\n\nV Magnitudes (Absolute):\n{v_magnitudes}")


# In[431]:


# Print problem statement
print('\n\nProblem 10')
print('Part a.)\n')

# Filter cluster 4 data
cluster4_data_filtered = cluster4_data[cluster4_data['_RAJ2000'] > 180]

# Define ages and metallicities for isochrones
ages = [7.0, 7.1, 7.2, 7.3]
metallicities = [0.0152]

# Define linestyles and colors for isochrones
linestyles = ['-', '--', '-.']
colors = ['b', 'r', 'm'] 

# Create a plot for the CMD of cluster 4
plt.figure(figsize=(8, 6))

# Plot the CMD for cluster 4
plt.scatter(
    cluster4_data_filtered['Bmag'] - cluster4_data_filtered['Vmag'],
    cluster4_data_filtered['Vmag'],
    c='k',  # Color for cluster 4 data points
    s=10,   # Marker size
    label='Cluster 4 CMD'
)

# Overlay isochrones
for age, linestyle, color in zip(ages, linestyles, colors):
    for metallicity in metallicities:
        # Filter isochrones
        isochrone = isochrones_data[
            (isochrones_data['logAge'] == age) & (isochrones_data['Zini'] == metallicity)
        ]
        # Plot the isochrones with age in millions of years in the legend
        plt.plot(
            isochrone['Bmag'] - isochrone['Vmag'],
            isochrone['Vmag'],
            linestyle=linestyle,
            color=color,
            label=f'Age: {age} logyr, Metallicity: {metallicity}'
        )

# Set labels and title
plt.xlabel('B - V')
plt.ylabel('V')
plt.title('CMD of Cluster 4 with Isochrones')

# Add legend
plt.legend()

# Invert the y-axis for the CMD of cluster 4
plt.gca().invert_yaxis()

# Show the plot
plt.show()


# In[432]:


# Answer questions in Midterm
print('\nWhat do you see?')
print('\nThe model isochrones do not align with the cluster data')
print('They only slightly overlap in the 5-15 V range.')

print('\n\nWhat can the sources of discrepancy between the model isochrone\n\
magnitudes and your CMD magnitudes be?')
print('\nSources of discrepancy could be the age and/or metallicity of the\n\
isochrones, as well as the distance to the cluster being incorrect,\n\
additional reddening of the sample, or incorrect calibration/isochrones.')


# In[441]:


# Print problem statement
print('\nPart b.)')

# Define the distance to cluster 4
distance_to_cluster4 = 1976  # in parsecs

# Correct the apparent magnitudes to absolute magnitudes
cluster4_data['MV'] = cluster4_data['Vmag'] - 5 * (np.log10(distance_to_cluster4) - 1)

# Define linestyles and colors for isochrones
linestyles = ['-', '--', '-.']
colors = ['b', 'r', 'm'] 

# Create a new plot
plt.figure(figsize=(8, 6))

# Plot the corrected CMD
plt.scatter(cluster4_data['Bmag'] - cluster4_data['Vmag'], cluster4_data['MV'], s=5)
plt.xlabel('B-V')
plt.ylabel('V')
plt.title('CMD of Cluster 4 (Corrected for Distance)')

# Invert the y-axis
plt.ylim(20,-10)

# Overlay isochrones
for age, linestyle, color in zip(ages, linestyles, colors):
    for metallicity in metallicities:
        # Filter isochrones
        isochrone = isochrones_data[
            (isochrones_data['logAge'] == age) & (isochrones_data['Zini'] == metallicity)
        ]
        # Plot the isochrones
        plt.plot(
            isochrone['Bmag'] - isochrone['Vmag'],
            isochrone['Vmag'],
            linestyle=linestyle,
            color=color,
            label=f'Age: {age}, Metallicity: {metallicity}'
        )
        
# Save the corrected plot as a PDF
plt.tight_layout()
plt.savefig('midterm_abigailglover_problem10_graph2.pdf')

# Show the corrected plot
plt.show()


# In[425]:


# Answer question from midterm
print('\n\nThis should have fixed the V mag problem. Why?\n')

print('This corrected the V mag values to align with the cluster\n\
because absolute magnitudes take into account the distance to the\n\
cluster (which affects how bright the stars appear). Because I was\n\
not originally accounting for this data, the isochrones were off.')


# In[442]:


# Print problem statement
print('\nPart c.')

# Define the assumed color excess and its uncertainty
color_excess = 0.36
color_excess_uncertainty = 0.03

# Correct the observed color for reddening
cluster4_data['B-V'] = cluster4_data['Bmag'] - cluster4_data['Vmag']
dereddened_color = cluster4_data['B-V'] - color_excess

# Define linestyles and colors for isochrones
linestyles = ['-', '--', '-.']
colors = ['b', 'r', 'm'] 

# Create a new plot
plt.figure(figsize=(8, 6))

# Plot the CMD with dereddened color
plt.scatter(dereddened_color, cluster4_data['MV'], s=5)
plt.xlabel('Dereddened B-V')
plt.ylabel('V')
plt.title('CMD of Cluster 4 ( Corrected for Reddening)')

# Invert the y-axis
plt.ylim(20,-10)

# Overlay isochrones
for age, linestyle, color in zip(ages, linestyles, colors):
    for metallicity in metallicities:
        # Filter isochrones
        isochrone = isochrones_data[
            (isochrones_data['logAge'] == age) & (isochrones_data['Zini'] == metallicity)
        ]
        # Plot the isochrones
        plt.plot(
            isochrone['Bmag'] - isochrone['Vmag'],
            isochrone['Vmag'],
            linestyle=linestyle,
            color=color,
            label=f'Age: {age}, Metallicity: {metallicity}'
        )
        
# Save the plot
plt.tight_layout()
plt.savefig('midterm_abigailglover_problem10_graph3.pdf')

# Show the plot
plt.show()


# In[427]:


# Answer questions in the midterm
print('\n\nThis should have fixed the B-V mag problem, too. Why?\n')

print('By subtracting the assumed color excess (E(B-V) = 0.36) from the\n\
observed B-V color, the data is effectively dereddened. This is needed because\n\
interstellar dust, which scatters and absorbs starlight as it passes through\n\
space causes stars to appear dimmer and redder than they actually are, causing\n\
the data to not be accurate off the bat.')


# In[444]:


# Print problem statement
print('\n\nProblem 11\n')
print('Part a.')

# Calculate the mean values
mean_B = cluster4_data_filtered['Bmag'].mean()
mean_V = cluster4_data_filtered['Vmag'].mean()

# Calculate the standard deviations
std_B = cluster4_data_filtered['Bmag'].std()
std_V = cluster4_data_filtered['Vmag'].std()

# Define the threshold (I chose 3 sigma)
threshold = 3

# Remove outliers based on the threshold
filtered_data = cluster4_data_filtered[
    (abs(cluster4_data_filtered['Bmag'] - mean_B) < threshold * std_B) &
    (abs(cluster4_data_filtered['Vmag'] - mean_V) < threshold * std_V)
]

# Create a new plot for the mean cluster CMD with outliers removed
plt.figure(figsize=(8, 6))

# Plot the mean cluster CMD with outliers removed
plt.scatter(filtered_data['Bmag'] - filtered_data['Vmag'], filtered_data['Vmag'], s=5)
plt.xlabel('B-V')
plt.ylabel('V mag')
plt.title('Mean Cluster CMD (Outliers Removed)')

# Invert the y-axis
plt.ylim(25,-10)

# Overlay isochrones
for age, linestyle, color in zip(ages, linestyles, colors):
    for metallicity in metallicities:
        isochrone = isochrones_data[
            (isochrones_data['logAge'] == age) & (isochrones_data['Zini'] == metallicity)
        ]
        plt.plot(
            isochrone['Bmag'] - isochrone['Vmag'],
            isochrone['Vmag'],
            linestyle=linestyle,
            color=color,
            label=f'Age: {age}, Metallicity: {metallicity}'
        )

# Save the plot
plt.tight_layout()
plt.savefig('midterm_abigailglover_problem11_graph1.pdf')

# Show the plot
plt.show()


# In[445]:


# Print problem statement
print('\nPart b.)\n')

# Create arrays to store the sum of squared differences for each isochrone
ssd_values = np.zeros((len(ages), len(metallicities)))

for i, age in enumerate(ages):
    for j, metallicity in enumerate(metallicities):
        isochrone = isochrones_data[
            (isochrones_data['logAge'] == age) & (isochrones_data['Zini'] == metallicity)
        ]

        # Calculate the predicted V mag values for the isochrone using Bmag and Vmag from filtered_data
        predicted_V = np.interp(filtered_data['Bmag'] - filtered_data['Vmag'], isochrone['Bmag'] - isochrone['Vmag'], isochrone['Vmag'])

        # Calculate residuals
        residuals = filtered_data['Vmag'] - predicted_V

        # Calculate the sum of squared differences
        ssd = np.sum(residuals**2)
        ssd_values[i][j] = ssd

# Find the indices of the minimum value in the array 
best_fit_indices = np.unravel_index(np.argmin(ssd_values), ssd_values.shape)

# Unpack the indices into two separate variables
best_age_index, best_metallicity_index = best_fit_indices

# Get the best-fitting age
best_age = ages[best_age_index]

# Get the best fitting metallicity
best_metallicity = metallicities[best_metallicity_index]

# Get the best fitting isochrone
best_fit_isochrone = isochrones_data[
    (isochrones_data['logAge'] == best_age) & (isochrones_data['Zini'] == best_metallicity)
]

# Create a new plot for the mean cluster CMD with outliers removed
plt.figure(figsize=(8, 6))

# Plot the mean cluster with outliers removed
plt.scatter(filtered_data['Bmag'] - filtered_data['Vmag'], filtered_data['Vmag'], s=5)
plt.xlabel('B-V')
plt.ylabel('V')
plt.title('Mean Cluster CMD with Best Fit Isochrones')

# Invert the y-axis
plt.ylim(25,-10)

# Plot the best-fitting isochrone
plt.plot(
    best_fit_isochrone['Bmag'] - best_fit_isochrone['Vmag'],
    best_fit_isochrone['Vmag'],
    linestyle='-',
    color='k',
    label=f'Best Fit - Age: {best_age}, Metallicity: {best_metallicity}'
)

# Save the plot
plt.tight_layout()
plt.savefig('midterm_abigailglover_problem11_graph2.pdf')

# Show the plot
plt.show()

# Answer question from midterm
print('\nWhat is the best fit age you find for cluster 4?')
print('\nBased on the fit of the data and the other ages tested that are not presently shown,\n\
an age of 7 log years was found to best represent the data. There was an issue with getting\n\
isochrones to show up at all on my plot, even after trying multiple age ranges that would more reasonably\n\
make sense with an open cluster. I believe this is due to my unfamiliarity with the isochrone database\n\
and not properly understanding how to navigate. After increasing the age SIGNIFICANTLY, the isochrones\n\
at least showed up in the plot. The findings, however, do not make sense for the cluster.')


# In[447]:


# Print problem statement
print('\n\nProblem 12')

# Specify the file path
file_path = 'omega_cen_isochrones.txt'

# Set the delimiter
delimiter = r'\s+'

# Read the data without specifying the header
isochrones_data = pd.read_csv(file_path, delimiter=delimiter, engine='python', comment='#', header=None)

# Manually specify column names
column_names = [
    "Zini", "MH", "logAge", "Mini", "int_IMF", "Mass", "logL", "logTe",
    "logg", "label", "McoreTP", "C_O", "period0", "period1", "period2", "period3",
    "period4", "pmode", "Mloss", "tau1m", "X", "Y", "Xc", "Xn", "Xo", "Cexcess", "Z",
    "mbolmag", "Umag", "Bmag", "Vmag", "Rmag", "Imag", "Jmag", "Hmag", "Kmag"
]

# Assign the column names to the DataFrame
isochrones_data.columns = column_names

# Remove # symbol from column names
isochrones_data.columns = isochrones_data.columns.str.strip('# ')

# Get ages and metallicities
ages = isochrones_data['logAge'].unique()
metallicities = isochrones_data['Zini'].unique()

# Print the extracted information
print("Unique Ages:", ages)
print("Unique Metallicities:", metallicities)

# Filter Omega Centauri data
omega_cen_data_filtered = omega_cen_data
# [omega_cen_data['RA'] > 180]

# Define linestyles and colors for isochrones
linestyles = ['-', '--', '-.']
colors = ['b', 'r', 'm'] 

# Create a plot for the CMD of Omega Centauri
plt.figure(figsize=(8, 6))

# Invert y-axis
plt.ylim(40,-5)

# Plot the CMD for Omega Centauri
plt.scatter(
    omega_cen_data_filtered['b[mag]'] - omega_cen_data_filtered['v[mag]'],
    omega_cen_data_filtered['v[mag]'],
    c='k',  # Color for Omega Centauri data points
    s=10,   # Marker size
    label='Omega Centauri CMD'
)

# Overlay isochrones
for age, linestyle, color in zip(ages, linestyles, colors):
    for metallicity in metallicities:
        # Filter isochrones
        isochrone = isochrones_data[
            (isochrones_data['logAge'] == age) & (isochrones_data['Zini'] == metallicity)
        ]
        # Plot the isochrones with age in billions of years (Byr) in the legend
        plt.plot(
            isochrone['Bmag'] - isochrone['Vmag'],
            isochrone['Vmag'],
            linestyle=linestyle,
            color=color,
            label=f'Age: {age} Byr, Metallicity: {metallicity}'
        )

# Show the plot
plt.show()


# In[434]:


print('\nNOTE: I ran into an issue with the isochrones for this problem as well\n\
where the isochrone data does not make sense with the cluster data. I had difficulty\n\
getting various age ranges and metallicities to show up at all on my plot due to\n\
a lack of knowledge on my part. However, I still tried to process the incorrect isochrones\n\
the same way I would have with the correct isochrones.')


# In[ ]:




