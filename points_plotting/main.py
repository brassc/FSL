# This python program imports .nii files to python, extracts a slice and prints it.

import nibabel as nib
import scipy as sp
import numpy as np
import pandas as pd
from PIL import Image

log_file="nifti_load_log.txt"

# Load the NIfTI image
try:
    with open(log_file, "w") as f:
        img = nib.load('/home/cmb247/Desktop/Project_3/BET_Extractions/19978/T1w_time1_registered_scans/T1w_time1.T1w_verio_P00030_19978_acute_20111102_U-ID22791_registered.nii.gz')
        print("NIfTI file loaded successfully.")

        # print information
        f.write("image shape: {}\n".format(img.shape))
        f.write("Affine transformation matrix:\n{}\n".format(img.affine))
        f.write("Header information:\n{}\n".format(img.header))

        data = img.get_fdata()
        f.write("Image data:\n{}\n".format(data))
    
except Exception as e:
        print("Error loading NIfTI file:", e)



# Get the affine transformation matrix
affine = img.affine

# Define the scanner coordinates or voxel location
scanner_coords = [2.641497, -2.877373, -12.73399]  
#voxel loc: 91 119 145

# Convert scanner coordinates to voxel indices
voxel_indices = np.linalg.inv(affine).dot([scanner_coords[0], scanner_coords[1], scanner_coords[2], 1])
voxel_indices = voxel_indices.astype(int)[:3]  # Extract integer voxel indices


# Extract the axial slice at the z voxel index determined from the scanner coordinates
z_index = voxel_indices[2]
slice_data = data[:, :, z_index]

# Normalize the slice data for image display
normalized_slice = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255

# Convert the normalized slice to a PIL image for saving or displaying
slice_image = Image.fromarray(normalized_slice.astype(np.uint8))

# Rotate 90 degrees to be expected orientation (i.e. rotate counter-clockwise)
rotated_slice_image = slice_image.rotate(90, expand=True)

rotated_slice_image.show()


