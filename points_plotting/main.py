# main.py program from which functions are called

# libraries
import nibabel as nib
import scipy as sp
import numpy as np
import pandas as pd
#from PIL import Image
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

#user defined functions
from load_nifti import load_nifti
from polynomial_plot import create_polynomial
from symmetry_line import get_mirror_line
from symmetry_line import reflect_across_line
#from extract_slice import extract_and_display_slice


poi_log_file_path='/home/cmb247/repos/FSL/points_plotting/points.csv'
baseline_poi_log_file_path='/home/cmb247/repos/FSL/points_plotting/baseline_points.csv'

poi_voxels_file_path='/home/cmb247/repos/FSL/points_plotting/points_voxel_coords.csv'
baseline_poi_voxels_file_path='/home/cmb247/repos/FSL/points_plotting/baseline_points_voxel_coords.csv'


nifti_file_path ='/home/cmb247/Desktop/Project_3/BET_Extractions/19978/T1w_time1_registered_scans/T1w_time1.T1w_verio_P00030_19978_acute_20111102_U-ID22791_registered.nii.gz'
 
img, save_directory = load_nifti(nifti_file_path)


# get the affine transformation matrix 
affine=img.affine

# Get image data
data = img.get_fdata()

## SELECT AND PLOT BASE SLICE

# Define the scanner (RAS, anatomical, imaging space) coordinates or voxel location
scanner_coords = np.array([2.641497, -2.877373, -12.73399,1])  
#voxel loc: 91 119 145

# Inverse affine to convert RAS/anatomical coords to voxel coords
inv_affine=np.linalg.inv(img.affine)

# convert RAS/anatomical coords to voxel coords
voxel_coords=inv_affine.dot(scanner_coords)[:3]

# Extract the axial slice at the z voxel index determined from the scanner coordinates
z_index=int(voxel_coords[2])
slice_data=data[:,:, z_index]

# plot the axial slice
plt.imshow(slice_data.T, cmap='gray', origin='lower')



## POINTS OF INTEREST
"""
poi_df=pd.read_csv(poi_log_file_path)

transformed_points = []
for index, row in poi_df.iterrows():
    point = np.array([row[0], row[1], row[2], 1])
    transformed_point = np.linalg.inv(affine).dot(point)
    transformed_points.append(transformed_point)

# extract x and y coordinatess
transformed_points=np.array(transformed_points)
x_coords = transformed_points[:, 0]
y_coords = transformed_points[:, 1]

# Mark the RAS/scanner points of interest on the slice
plt.scatter(x_coords, y_coords, c='red', s=2)
"""
## POLYNOMIAL FITTING

# Deformed side
poly_func, x_values, y_values, xa_coords, ya_coords = create_polynomial(poi_log_file_path, affine)

#Baseline side
polyb_func, xb_values, yb_values, xb_coords, yb_coords = create_polynomial(baseline_poi_log_file_path, affine)


# Plot the fitted polynomial curve
plt.plot(x_values, y_values, color='red', label='Fitted Polynomial')
plt.scatter(xa_coords, ya_coords, c='red', s=2)
plt.scatter(xb_coords, yb_coords, c='r', s=2)
plt.plot(xb_values, yb_values, color='red', label='Baseline Polynomial')





# FINDING MIRRORLINE OF SELECTED POINTS xa and xb
m, c, Y = get_mirror_line(yb_coords, xa_coords, xb_coords)

#extend Y fit line
y_values = np.linspace(Y[0]+50, Y[-1]-50, 100)

# Calculate the corresponding x values from linear regression model, Y
x_values = m * y_values + c

#REFLECT BASELINE POINTS
xr = reflect_across_line(m, c, xb_coords, yb_coords)


# plot points and lines
plt.plot(x_values, y_values, color='blue', label='Mirror') # plot mirror line
plt.scatter(xr, yb_coords, color='blue', s=2) # plot mirrored points

# Save plot and show
save_path=os.path.join(save_directory, 'slice_plot.png')
print('Plot saved to '+ save_path)
plt.savefig(save_path)
plt.show()




