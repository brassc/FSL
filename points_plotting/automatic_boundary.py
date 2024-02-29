import cv2
import numpy as np
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt

from make_patient_dir import ensure_directory_exists
from load_nifti import load_nifti
from load_np_data import load_data_readout


# loads or creates points directory path and associated points files based on patient ID and timepoint
patient_id='19978'
patient_timepoint='acute'
directory_path = ensure_directory_exists(patient_id, patient_timepoint)
nifti_file_path ='/home/cmb247/Desktop/Project_3/BET_Extractions/19978/T1w_time1_registered_scans/T1w_time1.T1w_verio_P00030_19978_acute_20111102_U-ID22791_registered.nii.gz'
slice_selected=np.array([2.641497, -2.877373, -12.73399,1]) # Scanner coordinates
#load mask created from slice_bet_script.sh
bet_mask_file_path="/home/cmb247/Desktop/Project_3/BET_Extractions/19978/T1w_time1_registered_scans/acute_restored_bet_mask-f0.5-R.nii.gz"

#${directory}${patient_timepoint}_restored_bet_mask${bet_p}




img, save_directory = load_nifti(nifti_file_path)

    # get the affine transformation matrix 
affine=img.affine

    # Get image data
data = img.get_fdata()

    ## SELECT

    # Define the scanner (RAS, anatomical, imaging space) coordinates or voxel location
scanner_coords = slice_selected 
    #voxel loc: 91 119 145

    # Inverse affine to convert RAS/anatomical coords to voxel coords
inv_affine=np.linalg.inv(img.affine)

    # convert RAS/anatomical coords to voxel coords
voxel_coords=inv_affine.dot(scanner_coords)[:3]

    # Extract the axial slice at the z voxel index determined from the scanner coordinates
z_index=int(voxel_coords[2])
slice_data=data[:,:, z_index]
slice_data=slice_data.T[::-1, :]

# Normalize the slice for better visualization (0-255 range)
normalized_slice = 255 * (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
normalized_slice = normalized_slice.astype(np.uint8)

#Save rotated image as jpg
directory_path='/home/cmb247/Desktop/Project_3/BET_Extractions/19978/T1w_time1_registered_scans/'
output_path=f"{directory_path}slice.png"
Image.fromarray(normalized_slice, 'L').save(output_path)



data_readout_loc = f"data_readout/{patient_id}_{patient_timepoint}"
polyd_func, x_values, y_values, xa_coords, ya_coords = load_data_readout(data_readout_loc, 'deformed_arrays.npz')
polyb_func, xb_values, yb_values, xb_coords, yb_coords = load_data_readout(data_readout_loc, 'baseline_arrays.npz')
polyr_func, xr_values, yr_values, xr_coords, yb_coords = load_data_readout(data_readout_loc, 'reflected_baseline_arrays.npz')
"""
plt.imshow(slice_data, cmap='gray')
plt.scatter(75, 100, c='red')
plt.scatter(xr_coords[0], yb_coords[-1], c='orange')
plt.scatter(xr_coords[-1], yb_coords[0])
plt.show()
"""

# Ensure the starting index is smaller than the ending index
start_y = int(min(yb_coords[-1], yb_coords[0]))
end_y = int(max(yb_coords[-1], yb_coords[0]))

# Slice slice_data between these y-coordinates
trimmed_slice_data = slice_data[start_y:end_y, 120:]
"""
plt.imshow(trimmed_slice_data, cmap='gray')
plt.show()
"""

#normalise trimmed data 
norm_tr_slice = 255 * (trimmed_slice_data - np.min(trimmed_slice_data)) / (np.max(trimmed_slice_data) - np.min(trimmed_slice_data))
norm_tr_slice = norm_tr_slice.astype(np.uint8)

# Apply Canny edge detection to find edges in the image
edges = cv2.Canny(norm_tr_slice, 100, 200)  # You can adjust thresholds as needed

# Find contours from the detected edges
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty image to draw contours
contour_img = np.zeros_like(norm_tr_slice)

# Draw contours on the image
cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 1)
"""
# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(contour_img, cmap='gray')
plt.title('Brain Boundary')
plt.show()
"""

mask_nifti = nib.load(bet_mask_file_path)

# Step 2: Access the image data
mask_data = mask_nifti.get_fdata()


# Step 3: Check for binary values
unique_values = np.unique(mask_data)
print("Unique values in the mask:", unique_values)

# Verify it's binary
if np.array_equal(unique_values, [0, 1]) or np.array_equal(unique_values, [0]) or np.array_equal(unique_values, [1]):
    print("The mask is binary.")
else:
    print("The mask is not strictly binary.")

# Step 4: Visual inspection
# Choose a slice index
slice_index = 145  # Middle slice, adjust as needed

transposed_slice=np.transpose(mask_data[:,:,slice_index])
corrected_slice=transposed_slice#np.flipud(transposed_slice)

# Plot the slice
plt.imshow(corrected_slice, cmap='gray')
plt.title(f'Slice at index {slice_index}')
# Adjust the y-axis to display in the original image's orientation
plt.gca().invert_yaxis()
plt.show()

# Ensure the starting index is smaller than the ending index
start_y = int(min(yb_coords[-1], yb_coords[0]))
end_y = int(max(yb_coords[-1], yb_coords[0]))
x_offset = 120  # Adjust based on your earlier trimming

# Slice corrected_slice between these y-coordinates
trimmed_slice_data = corrected_slice[start_y:end_y, x_offset:]
plt.imshow(trimmed_slice_data, cmap='gray')
# Adjust the y-axis to display in the original image's orientation
plt.gca().invert_yaxis()
plt.show()

# Assume corrected_slice has the original dimensions, e.g., from a 256x256 slice
original_shape = corrected_slice.shape

# Create a zero-filled array with the same dimensions as the original slice
restored_slice = np.zeros(original_shape)


# Insert the trimmed data back into the restored_slice at the original position
end_y = start_y + trimmed_slice_data.shape[0]  # Calculated based on the trimmed data size
end_x = x_offset + trimmed_slice_data.shape[1]  # Calculated based on the trimmed data size

restored_slice[start_y:end_y, x_offset:end_x] = trimmed_slice_data

# Display the restored slice such that trimmed area fills the plot
# You can plot this data so it fills the plot but maintains its reference to the original coordinate system
plt.imshow(trimmed_slice_data, cmap='gray', extent=[120, 120 + trimmed_slice_data.shape[1], end_y, start_y])

# Adjust the y-axis to display in the original image's orientation
plt.gca().invert_yaxis()

# Labeling for context
plt.xlabel('X coordinate in original image')
plt.ylabel('Y coordinate in original image')
plt.title('Trimmed Slice Displayed in Original Coordinates')
plt.scatter(xa_coords, ya_coords)
plt.scatter(xr_coords, yb_coords, s=2)

plt.show()

"""
# Load the NIfTI mask file
bet_mask = nib.load(bet_mask_file_path)
# get affine and mask data
affine_mask=bet_mask.affine
mask_data=bet_mask.get_fdata()
mask_data_binary=mask_data.astype(bool)

# apply affine transformation to get voxel coordinates
inv_mask_affine = np.linalg.inv(affine_mask)
mask_voxel_coords = inv_mask_affine.dot(scanner_coords)[:3]
z_index_mask = int(mask_voxel_coords[2])

# Extract the specific slice data
mask_slice_data = mask_data[:, :, z_index_mask]
# reorient correctly
mask_slice_data=mask_slice_data.T[::-1,:]
"""
"""
# Normalize the slice data to the range [0, 255]
mask_normalized_slice = 255 * (mask_slice_data - np.min(mask_slice_data)) / (np.max(mask_slice_data) - np.min(mask_slice_data))
mask_normalized_slice = mask_normalized_slice.astype(np.uint8)
"""
"""

plt.imshow(mask_slice_data_binary, cmap='gray')
plt.show()

"""







