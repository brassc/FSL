# main.py program from which functions are called

# libraries
import nibabel as nib
import scipy as sp
import numpy as np
import pandas as pd
#from PIL import Image
import matplotlib.pyplot as plt
import os

#user defined functions
from load_nifti import load_nifti
from polynomial_plot import create_polynomial
#from extract_slice import extract_and_display_slice


poi_log_file_path='/home/cmb247/repos/FSL/points_plotting/points.csv'
poi_df=pd.read_csv(poi_log_file_path)

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

transformed_points = []
for index, row in poi_df.iterrows():
    point = np.array([row[0], row[1], row[2], 1])
    transformed_point = np.linalg.inv(affine).dot(point)
    transformed_points.append(transformed_point)

# extract x and y coordinatess
transformed_points=np.array(transformed_points)
x_coords = transformed_points[:, 0]
y_coords = transformed_points[:, 1]
#z_coords = transformed_points[:, 2]

#print(x_coords)
#print(y_coords)





# Mark the RAS/scanner points of interest on the slice
plt.scatter(x_coords, y_coords, c='red', s=2)

## POLYNOMIAL FITTING
"""
# Fit a polynomial of degree 2 relating x to y
coefficients = np.polyfit(y_coords, x_coords, 2)

# Create a polynomial function using the coefficients
poly_func = np.poly1d(coefficients)

# Print the polynomial equation
print("Polynomial Equation:")
print(poly_func)

# Generate points for the fitted polynomial curve
poi_df_max_index=np.size(x_coords)-1
y_values = np.linspace(y_coords[0],y_coords[poi_df_max_index], 100)
x_values = poly_func(y_values)
"""

poly_func, x_values, y_values = create_polynomial(poi_log_file_path, affine)

# Plot the fitted polynomial curve
plt.plot(x_values, y_values, color='red', label='Fitted Polynomial')


# Save plot
save_path=os.path.join(save_directory, 'slice_plot.png')
print('Plot saved to '+ save_path)

plt.show()
















#normalized_slice = extract_and_display_slice(nifti_file_path, scanner_coords)



#plt.imshow(img.T, cmap='gray', origin='lower')
#plt.show()

"""
# AFFINE TRANSFORMATION - convert voxel coordinates to image space coordinates
# For a voxel at position (x, y, z) in the data array
voxel_coords = [x, y, z, 1]

# Convert voxel coordinates to world coordinates
world_coords = img.affine.dot(voxel_coords)[:3]

"""


"""
# Convert the normalized slice to a PIL image for saving or displaying
slice_image = Image.fromarray(normalized_slice.astype(np.uint8))

# Rotate 90 degrees to expected orientation (i.e., rotate counter-clockwise)
rotated_slice_image = slice_image.rotate(90, expand=True)
rotated_slice_image.show()

"""






