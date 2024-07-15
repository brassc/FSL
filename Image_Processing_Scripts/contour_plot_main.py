import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import nibabel as nib
from PIL import Image


# userdef functions
# THIS FUNCTION LOADS A NIFTI FILE AND RETURNS IT 
def load_nifti(nifti_file_path):
    # Extract the directory from nifti_file_path
    nifti_directory = os.path.dirname(nifti_file_path)

    # Define the directory for saving the image
    save_directory = os.path.join(nifti_directory, 'contour_plots')
    
    # Check if the save_directory exists, if not, create it
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Define the log file path inside the save_directory
    log_file = os.path.join(save_directory, "nifti_load_log.txt")

    try:
        with open(log_file, "w") as f:
            img = nib.load(nifti_file_path)
            print("NIfTI file loaded successfully.")

            # Print information
            f.write(f"image shape: {img.shape}\n")
            f.write(f"Affine transformation matrix:\n{img.affine}\n")
            f.write(f"Header information:\n{img.header}\n")

            data = img.get_fdata()
            f.write(f"Image data:\n{data}\n")
    
    except Exception as e:
        print(f"Error loading NIfTI file: {e}")
        return
    
    return img, save_directory

# THIS FUNCTION EXTRACTS AND DISPLAYS A SLICE FROM A NIFTI FILE
# to do: decide what voxel_indices looks like
def extract_and_display_slice(nifti_file_path, voxel_indices):

    img, save_directory = load_nifti(nifti_file_path)

    # Get the affine transformation matrix
    affine = img.affine

    ## Convert scanner coordinates to voxel indices
    #voxel_indices = np.linalg.inv(affine).dot([scanner_coords[0], scanner_coords[1], scanner_coords[2], 1])
    #voxel_indices = voxel_indices.astype(int)[:3]  # Extract integer voxel indices

    # Extract the axial slice at the z voxel index determined from the scanner coordinates
    z_index = voxel_indices[2]
    slice_data = data[:, :, z_index]

    # Normalize the slice data for image display
    normalized_slice = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255

    # Convert the normalized slice to a PIL image for saving or displaying
    slice_image = Image.fromarray(normalized_slice.astype(np.uint8))

    # Rotate 90 degrees to expected orientation (i.e., rotate counter-clockwise)
    rotated_slice_image = slice_image.rotate(90, expand=True)

    # Save and display the image
    save_path=os.path.join(save_directory, 'slice_extraction.png')
    rotated_slice_image.save(save_path)  # Save the corrected slice image
    rotated_slice_image.show()

    return normalized_slice




# maybe this function is unnecessary......
def load_boundary_detection_features(patient_id, patient_timepoint, bet_mask_file_path):
    directory_path = ensure_directory_exists(patient_id, patient_timepoint)
    data_readout_loc = f"data_readout/{patient_id}_{patient_timepoint}"
    xa_coords, ya_coords = load_auto_data_readout(data_readout_loc, 'auto_deformed_array.npz')
    xb_coords, yb_coords = load_auto_data_readout(data_readout_loc, 'auto_baseline_array.npz')
    xr_coords, yb_coords = load_auto_data_readout(data_readout_loc, 'auto_reflected_array.npz')

        
    mask_nifti = nib.load(bet_mask_file_path)

    # Step 2: Access the image data
    mask_data = mask_nifti.get_fdata()

    # Step 3: Check for binary values
    unique_values = np.unique(mask_data)
    print("Unique values in the mask:", unique_values)

    # Verify it's binary
    if np.array_equal(unique_values, [0, 1]) or np.array_equal(unique_values, [0]) or np.array_equal(unique_values, [1]):
        print("The mask is binary.")
        mask_data = mask_data  # No change needed; just assign it directly
    else:
        print("The mask is not strictly binary. Binarizing now...")
        # Correct binarization: Apply the condition to the entire mask_data array
        mask_data = (mask_data > 0).astype(np.uint8)
    
    # Step 4: Visual inspection - plot at chosen slice index
    file_loc = f"points_dir/{patient_id}_{patient_timepoint}/points_voxel_coords.csv"
    # Load the CSV file using pandas
    df = pd.read_csv(file_loc)
    # Retrieve the slice index value assuming it's stored in the second row and fourth column (1,3 in 0-indexed)
    print(df.head(2))
    slice_index=int(df.iloc[0, 2])

    corrected_slice=np.transpose(mask_data[:,:,slice_index])

    # Plot the entire slice
    plt.imshow(corrected_slice, cmap='gray')
    plt.title(f'Slice at index {slice_index}')
    # Adjust the y-axis to display in the original image's orientation
    plt.gca().invert_yaxis()
    plt.show()

    return corrected_slice, xa_coords, ya_coords, xb_coords, yb_coords, xr_coords


# THIS FUNCTION OUTPUTS CONTOUR X AND Y COORDINATES
# Edit what it takes in and why
def auto_boundary_detect(patient_id, patient_timepoint, bet_mask_file_path, x_offset, array_save_name):
    
    corrected_slice, xa_coords, ya_coords, xb_coords, yb_coords, xr_coords = load_boundary_detection_features(patient_id, patient_timepoint, bet_mask_file_path)

    # PLOT REGION ONLY BASED ON x_offset VALUE 

    # Ensure the starting index is smaller than the ending index
    start_y = int(min(yb_coords[-1], yb_coords[0]))
    end_y = int(max(yb_coords[-1], yb_coords[0]))
    
    
    if x_offset > 0.5 * corrected_slice.shape[1]:
        trimmed_slice_data = corrected_slice[start_y:end_y, x_offset:]
    else:
        trimmed_slice_data = corrected_slice[start_y:end_y, :x_offset]
    # Slice 'corrected_slice' between these y-coordinates and plot
    #trimmed_slice_data = corrected_slice[start_y:end_y, x_offset:]
    """
    plt.imshow(trimmed_slice_data, cmap='gray')
    # Adjust the y-axis to display in the original image's orientation
    plt.gca().invert_yaxis()
    plt.show()
    """

    # Assume corrected_slice has the original dimensions, e.g., from a 256x256 slice
    original_shape = corrected_slice.shape

    # Create a zero-filled array with the same dimensions as the original slice
    restored_slice = np.zeros(original_shape)


    # Insert the trimmed data back into the restored_slice at the original position
    end_y = start_y + trimmed_slice_data.shape[0]  # Calculated based on the trimmed data size
    end_x = x_offset + trimmed_slice_data.shape[1]  # Calculated based on the trimmed data size

    restored_slice[start_y:end_y, x_offset:end_x] = trimmed_slice_data

    """
    # Display the restored slice such that trimmed area fills the plot
    # You can plot this data so it fills the plot but maintains its reference to the original coordinate system
    if x_offset > 0.5 * corrected_slice.shape[1]:
        plt.imshow(trimmed_slice_data, cmap='gray', extent=[x_offset, x_offset + trimmed_slice_data.shape[1], end_y, start_y])
    else:
        plt.imshow(trimmed_slice_data, cmap='gray', extent=[x_offset - trimmed_slice_data.shape[1], x_offset, end_y, start_y])

    # Adjust the y-axis to display in the original image's orientation
    plt.gca().invert_yaxis()

    # Labeling for context
    plt.xlabel('X coordinate in original image')
    plt.ylabel('Y coordinate in original image')
    plt.title('Trimmed Slice Displayed in Original Coordinates')
    if x_offset > 0.5 * corrected_slice.shape[1]:
        plt.scatter(xa_coords, ya_coords, s=2, color='red')
        plt.scatter(xr_coords, yb_coords, s=2, color='cyan')
    else:   
        plt.scatter(xb_coords, yb_coords, s=2, color='cyan')
        
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
    # Adjust the y-axis to display in the original image's orientation
    plt.gca().invert_yaxis()
    plt.title('Brain Boundary')

    plt.show()
    """

    # Create an image based on the original image dimensions to position the contours correctly
    contour_img_original_ref = np.zeros(original_shape)
    # Insert the trimmed data back into the restored_slice at the original position
    end_y = start_y + contour_img.shape[0]  # Calculated based on the trimmed data size
    end_x = x_offset + contour_img.shape[1]  # Calculated based on the trimmed data size

    contour_img_original_ref[start_y:end_y, x_offset:end_x] = contour_img

    if x_offset > 0.5 * corrected_slice.shape[1]:
        plt.imshow(contour_img, cmap='gray', extent=[x_offset, x_offset + contour_img.shape[1], end_y, start_y])
    else:
        plt.imshow(contour_img, cmap='gray', extent=[x_offset - contour_img.shape[1], x_offset, end_y, start_y])

    # Adjust the y-axis to display in the original image's orientation
    plt.gca().invert_yaxis()

    # Labeling for context
    plt.xlabel('X coordinate in original image')
    plt.ylabel('Y coordinate in original image')
    plt.title('Trimmed Slice Boundary Displayed in Original Coordinates')
    if x_offset > 0.5 * corrected_slice.shape[1]:
        plt.scatter(xa_coords, ya_coords, s=2, color='red')
        plt.scatter(xr_coords, yb_coords, s=2, color ='cyan')
    else:   
        plt.scatter(xb_coords, yb_coords, s=2, color='cyan')

    plt.show()

    #GET POINTS IN ARRAY
    # Assuming 'contours' is obtained from cv2.findContours as per your provided code
    # Initialize an empty list to collect all points
    contour_points = []

    # Iterate through each contour
    for contour in contours:
        # Contour is an array of shape (n, 1, 2) where n is the number of points in the contour
        # We reshape the contour to shape (n, 2) and append to all_points list
        for point in contour.reshape(-1, 2):
        
            # Adjust the x coordinate
            adjusted_x = point[0] + x_offset
            # Adjust the y coordinate - note that y-coordinates need to consider the image's orientation
            adjusted_y = point[1] + start_y
            contour_points.append([adjusted_x, adjusted_y])


    # Convert the list of points to a NumPy array for easier manipulation and access
    contour_points_array = np.array(contour_points)
    contour_x_coords = contour_points_array[:,0]
    contour_y_coords = contour_points_array[:,1]

    # Save np arrays to to file.npz in given directory data_readout_dir using np.savez
    data_readout_dir=f"data_readout/{patient_id}_{patient_timepoint}"
    save_arrays_to_directory(data_readout_dir, array_save_name,
                                xx_coords=contour_x_coords, yy_coords=contour_y_coords)

    #no if statement necessary here because points are already adjustd
    plt.imshow(contour_img, cmap='gray', extent=[x_offset, x_offset + contour_img.shape[1], end_y, start_y])
    # Adjust the y-axis to display in the original image's orientation
    plt.gca().invert_yaxis()
    if x_offset > 0.5 * corrected_slice.shape[1]:
        plt.scatter(contour_x_coords, contour_y_coords, s=2, color='red')
    else:
        plt.scatter(contour_x_coords, contour_y_coords, s=2, color='cyan')
    plt.show()

    return contour_x_coords, contour_y_coords





# main script execution

# import patient info .csv
patient_info = pd.read_csv('Image_Processing_Scripts/included_patient_info.csv')
print(patient_info.head())


#bet_mask_file_path="/home/cmb247/Desktop/Project_3/BET_Extractions/"+str(patient_id)+"/T1w_time1_registered_scans/acute_restored_bet_mask-f0.5-R.nii.gz"


