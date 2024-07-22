import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import glob
import os
import sys
import nibabel as nib
from PIL import Image
from sklearn.linear_model import LinearRegression


# userdef functions

# USAGE
def print_usage():
    print("Usage: python contour_plot_main.py <patient_id>")
    print("If no patient_id is provided, the script will plot contours for all patient timepoints.")


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
# RETURNS SLICE IMAGE IN CORRECTED ORIENTATION
def extract_and_display_slice(img, save_directory, patient_id, timepoint, z_coord, disp_flag='y'):

    #img, save_directory = load_nifti(nifti_file_path)

    # Get the affine transformation matrix
    affine = img.affine

    ## Convert scanner coordinates to voxel indices
    #voxel_indices = np.linalg.inv(affine).dot([scanner_coords[0], scanner_coords[1], scanner_coords[2], 1])
    #voxel_indices = voxel_indices.astype(int)[:3]  # Extract integer voxel indices

    # Extract the axial slice at the z voxel index determined from the scanner coordinates
    data = img.get_fdata()
    slice_data = data[:, :, z_coord]

    # Normalize the slice data for image display - RETURN NORMALISED SLICE NP ARRAY
    normalized_slice = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
    rotated_norm_slice=np.rot90(normalized_slice, k=3) # rotate 90 degrees three times (k=3)
    oriented_norm_slice = np.fliplr(rotated_norm_slice)
    """
    plt.imshow(oriented_norm_slice, cmap='gray')
    ## Adjust the y-axis to display in the original image's orientation
    plt.gca().invert_yaxis()
    plt.show()
    """
    

    # FOR DISPLAY PURPOSES
    # Convert the normalized slice to a PIL image for saving or displaying
    slice_image = Image.fromarray(normalized_slice.astype(np.uint8))

    # Rotate 90 degrees to expected orientation (i.e., rotate counter-clockwise)
    rotated_slice_image = slice_image.rotate(270, expand=True)
    # Mirror this rotated image
    adjusted_slice_image = rotated_slice_image.transpose(Image.FLIP_LEFT_RIGHT)

    # Save and optionally display the image
    save_path=os.path.join(save_directory, 'slice_extraction.png')
    adjusted_slice_image.save(save_path)  # Save the corrected slice image
    
    if disp_flag=='y' or disp_flag=='yes':
        plt.imshow(adjusted_slice_image, cmap='gray', origin='lower')
        plt.title(f"Slice {z_coord} for patient {patient_id} at {timepoint}")
        #plt.axis('off')
        plt.show()
    elif disp_flag=='n' or disp_flag=='no':
         return oriented_norm_slice, adjusted_slice_image
    else:
        print("ERROR: type 'y', 'yes', 'n' or 'no' to state whether to display slice. Default is 'y'")
        sys.exit(1)

    return oriented_norm_slice, adjusted_slice_image

# THIS FUNCTION FLIPS 'SIDE' AND RETURNS IT AS 'FLIPSIDE'
def flipside_func(side):
    if side == 'R':
        flipside = 'L'
    elif side == 'L':
        flipside = 'R'
    else:
        print("ERROR: side must be 'R' or 'L'")
        sys.exit(1)
    return flipside

# THIS FUNCTION USES OPENCV TO DETECT BOUNDARY CONTOURS IN A GIVEN SLICE
# INPUT REQUIRED IS THE SLICE IMAGE, PATIENT INFO + COORDS
# RETURNS CONTOUR X AND Y COORDINATES
def auto_boundary_detect(patient_id, patient_timepoint, normalized_slice, antx, anty, postx, posty, side):

    # STEP 1: EXTRACT AND PLOT REGION OF INTEREST ONLY (RETAIN ORIGINAL COORDINATE SYSTEM)

    # Ensure the starting index is smaller than the ending index
    start_y = posty
    end_y = anty
    
    image_center_x = int(0.5 * normalized_slice.shape[1]) # work with np style nii slice
    
    if side == 'R':
        #crop_box = (0, start_y, image_center_x, end_y)
        trimmed_slice_data = normalized_slice[start_y:end_y, :image_center_x]        

    else:
        trimmed_slice_data = normalized_slice[start_y:end_y, image_center_x:]
    """    
    plt.imshow(trimmed_slice_data, cmap='gray')
    ## Adjust the y-axis to display in the original image's orientation
    plt.gca().invert_yaxis()
    plt.show()
    """
    
    # Assume adjusted_slice_image has the original dimensions, e.g., from a 256x256 slice
    original_shape = normalized_slice.shape

    # Create a zero-filled array with the same dimensions as the original slice
    restored_slice = np.zeros(original_shape)
    """
    print(f"trimmed slice data x shape: {trimmed_slice_data.shape[1]}")
    print(f"original x shape: {restored_slice.shape[1]}")
    print(f"image_center_x: {image_center_x}")
    """
    # Insert the trimmed data back into the restored_slice at the original position
    end_y = start_y + trimmed_slice_data.shape[0]  # Calculated based on the trimmed data size
    if side == 'R':
        start_x = 0
        end_x = image_center_x #+ trimmed_slice_data.shape[1]  # Calculated based on the trimmed data size
    else:
        start_x = image_center_x
        end_x = image_center_x + trimmed_slice_data.shape[1] 
    

    restored_slice[start_y:end_y, start_x:end_x] = trimmed_slice_data
    """
    plt.imshow(restored_slice, cmap='gray')
    ## Adjust the y-axis to display in the original image's orientation
    plt.gca().invert_yaxis()
    plt.show()
    """    

    """
    # Display the restored slice such that trimmed area fills the plot
    # You can plot this data so it fills the plot but maintains its reference to the original coordinate system
    plt.imshow(trimmed_slice_data, cmap='gray', extent=[start_x, end_x, end_y, start_y])
    
    # Adjust the y-axis to display in the original image's orientation
    plt.gca().invert_yaxis()

    # Labeling for context
    plt.xlabel('X coordinate in original image')
    plt.ylabel('Y coordinate in original image')
    plt.title('Trimmed Slice Displayed in Original Coordinates')
    plt.scatter(antx, anty, s=2, color='red')
    plt.scatter(postx, posty, s=2, color='cyan')
    
    plt.show()
    """

    # STEP 2: FIND CONTOURS
    #normalise trimmed data
    try: 
        norm_tr_slice = 255 * (trimmed_slice_data - np.min(trimmed_slice_data)) / (np.max(trimmed_slice_data) - np.min(trimmed_slice_data))
    except ValueError as e:
        print(f"Error processing slice: {e}")
        print(f"Skipping {patient_id} {timepoint} and continuing with the remaining scans.")
        if not np.any(trimmed_slice_data):
            print("Warning: Empty or zero-sized trimmed_slice_data array encountered.")
        return None, None
    


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

    contour_img_original_ref[start_y:end_y, start_x:end_x] = contour_img
    """
    if side == 'R':
        plt.imshow(contour_img, cmap='gray', extent=[start_x, end_x, end_y, start_y])
    else:
        plt.imshow(contour_img, cmap='gray', extent=[start_x, end_x, end_y, start_y])


    # Adjust the y-axis to display in the original image's orientation
    plt.gca().invert_yaxis()

    # Labeling for context
    plt.xlabel('X coordinate in original image')
    plt.ylabel('Y coordinate in original image')
    plt.title('Trimmed Slice Boundary Displayed in Original Coordinates')
    plt.show()
    """
    
    #GET POINTS IN ARRAY
    
    # Initialize an empty list to collect all points
    contour_points = []

    # Iterate through each contour
    for contour in contours:
        # Contour is an array of shape (n, 1, 2) where n is the number of points in the contour
        # We reshape the contour to shape (n, 2) and append to all_points list
        for point in contour.reshape(-1, 2):
        
            # Adjust the x coordinate
            adjusted_x = point[0] + start_x
            # Adjust the y coordinate - note that y-coordinates need to consider the image's orientation
            adjusted_y = point[1] + start_y
            contour_points.append([adjusted_x, adjusted_y])


    # Convert the list of points to a NumPy array for easier manipulation and access
    contour_points_array = np.array(contour_points)
    contour_x_coords = contour_points_array[:,0]
    contour_y_coords = contour_points_array[:,1]
    """
    # Save np arrays to to file.npz in given directory data_readout_dir using np.savez
    data_readout_dir=f"data_readout/{patient_id}_{patient_timepoint}"
    save_arrays_to_directory(data_readout_dir, array_save_name,
                                xx_coords=contour_x_coords, yy_coords=contour_y_coords)
    """
    """
    #no if statement necessary here because points are already adjustd
    plt.imshow(contour_img, cmap='gray', extent=[start_x, end_x, end_y, start_y])
    # Adjust the y-axis to display in the original image's orientation
    plt.gca().invert_yaxis()
    
    plt.scatter(contour_x_coords, contour_y_coords, s=2, color='red')
    plt.show()
    """    
    return contour_x_coords, contour_y_coords


def get_mirror_line(y_coords, xa_coords, xb_coords):
    #where b is baseline and a is expansion side
    # returns gradient, m; x intercept, c; fit data, Y

    # get first and last coordinates of contours
    first_avg_x = (xa_coords[0] + xb_coords[0]) / 2
    last_avg_x = (xa_coords[-1] + xb_coords[-1]) / 2
    #first_avg_x = xb_coords[0]
    #last_avg_x = xb_coords[-1]
    #print("xB_coords[0]", xb_coords[0])
    #print("xB_coords[-1]", xb_coords[-1])
    #print("y coords is:", y_coords)
    
    # Select the top and bottom in range of y_coords
    y_min = min(y_coords)
    y_max = max(y_coords)

    #print(f"top line point is: ({last_avg_x},{y_min})")
    #print(f"bottom line point is: ({first_avg_x},{y_max})")

    # Prepare data for regression
    X = np.array([last_avg_x, first_avg_x]).reshape(-1, 1)  # Dependent variable
    Y = np.array([y_min, y_max]).reshape(-1, 1)          # Independent variable
    print("X is: ",X)
    print("Y is: ", Y)
    # Initialize the linear regression model
    model = LinearRegression()

    # Fit the model to your data
    model.fit(X, Y)

    # The slope (gradient m) and intercept (c) from the fitted model
    m = model.coef_[0][0]
    if m == 0:
        c = first_avg_x
        print("No gradient found.")
        print('x intercept (c) is:', c)
        print("Performing simple reflection...")
    else:
        c = model.intercept_[0]

    print('Gradient (m) is:', m)
    print('x intercept (c) is:', c)

    return m, c, Y




def reflect_across_line(m, c, xb_coords, yb_coords):
    
    # if no gradient, then simple reflection (recall x and y directions are reversed...)
    if m == 0:
        yr = yb_coords
        xr = (c-xb_coords) + c
        return xr, yr

    # if gradient present, then do 2D reflection
    print("performing 2D reflection...")
    
    # number of points to reflect
    n = len(xb_coords)
    
    # Allocate space for new coords
    xr = np.zeros(n)
    yr = np.zeros(n)   

    for i in range(n):
        x = xb_coords[i]
        y = yb_coords[i]

        # Calculating the intersection point
        # Solve for x_i: (y +(1/m) * x - c) / (m + 1/m) = x_i

        denom = m + (1/m)
        x_i = (y + (1/m) * x - c) / denom
        
        y_i = m * x_i + c

        # Calculating the reflected point
        xr[i] = 2 * x_i - x
        yr[i] = 2 * y_i - y
   
    print("reflection complete.")
    return xr, yr



# THIS FUNCTION SEARCHES FOR A BET MASK FILE IN A GIVEN DIRECTORY
# RETURNS FILE PATH AS A LIST OF STRINGS (uses glob)
def search_for_bet_mask(directory, timepoint):
    # Construct the search pattern
    pattern = f"*{timepoint}*_bet_mask**.nii.gz"
    # Search for files matching the pattern in the specified directory
    files = glob.glob(os.path.join(directory, pattern))
    return files





# main script execution

# DATA CLEANING
# import patient info .csv
patient_info = pd.read_csv('Image_Processing_Scripts/included_patient_info.csv')

# Convert only numeric columns to integers
numeric_cols = patient_info.select_dtypes(include=['number']).columns
patient_info[numeric_cols] = patient_info[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)


# Strip leading and trailing spaces from the column names
patient_info.columns = patient_info.columns.str.strip()

# Verify the exact column names
print("Columns in the DataFrame:", patient_info.columns.tolist())

# Strip leading and trailing spaces from the data in each column
for col in patient_info.select_dtypes(include=['object']).columns:
    patient_info[col] = patient_info[col].str.strip()

# Rename columns to remove any hidden issues
patient_info.rename(columns={
    'patient ID': 'patient_id',
    'z coord (slice)': 'z_coord_slice',
    'anterior x coord': 'anterior_x_coord',
    'anterior y coord': 'anterior_y_coord',
    'posterior x coord': 'posterior_x_coord',
    'posterior y coord': 'posterior_y_coord',
    'side (L/R)': 'side_LR',
    'baseline anterior x coord': 'baseline_anterior_x_coord',
    'baseline posterior x coord': 'baseline_posterior_x_coord',
    'excluded?': 'excluded'
}, inplace=True)

# Remove excluded patients, remove this column
patient_info = patient_info[patient_info['excluded'] != 1]
patient_info.drop(columns='excluded', inplace=True)

#print(patient_info)

#filter for patient if desired
#patient_info = patient_info[patient_info['patient_id'] == 23348]

# Check if the user provided an argument (patient_id)
if len(sys.argv) > 1:
    try:
        patient_id = int(sys.argv[1])  # Convert argument to an integer
        # Filter the data for the specified patient_id
        patient_info = patient_info[patient_info['patient_id'] == patient_id]
        print(f"Plotting contours for patient_id {patient_id}")
    except ValueError:
        print("Invalid patient_id provided. It must be an integer.")
        print_usage()
        sys.exit(1)
else:
    print("No patient_id provided. Default will plot contours for all patient timepoints.")
    print_usage()
    input("Press any key to continue... (CTRL+C to cancel)")


# Iterate over each patient and timepoint
for patient_id, timepoint in zip(patient_info['patient_id'], patient_info['timepoint']):
    # Define the bet_mask_file_path for each patient and timepoint
    directory = f"/home/cmb247/Desktop/Project_3/BET_Extractions/{patient_id}/T1w_time1_bias_corr_registered_scans/BET_Output"
    
    # SEARCH FOR FILEPATH
    # Construct the search pattern - broad pattern first
    # Define broader patterns
    broad_pattern = f"*{timepoint}*_bet*.nii.gz"
    broad_pattern_priority = f"*{timepoint}*_bet*modified*.nii.gz"

    pattern = f"*{timepoint}*_bet_mask*.nii.gz"
    pattern_priority = f"*{timepoint}*_bet_mask*modifiedmask*.nii.gz"
    # Search for bet files matching the pattern, excluding 'mask' files in the specified directory
    img_filepath = [file for file in glob.glob(os.path.join(directory, broad_pattern_priority)) if 'mask' not in file]
    if not img_filepath:
        img_filepath = [file for file in glob.glob(os.path.join(directory, broad_pattern)) if 'mask' not in file]

    if img_filepath:
        if timepoint=='fast':
            filtered_paths = [path for path in img_filepath if "fast" in path and "ultra-fast" not in path]
            if filtered_paths:
                img_filepath=filtered_paths[0]
        else:
            img_filepath = img_filepath[0] # glob returns a list. this gets first element of list
        print (f"timepoint: {timepoint} \nfilepath: {img_filepath}")
    else:
        print("No bet file found for patient_id", patient_id, "timepoint", timepoint, ". Using mask instead...")
    

    # Repeat for finding mask using pattern and pattern priority
    mask_filepath = glob.glob(os.path.join(directory, pattern_priority))
    if not mask_filepath:
        mask_filepath = glob.glob(os.path.join(directory, pattern))

    if mask_filepath:
        if timepoint=='fast':
            filtered_paths = [path for path in mask_filepath if "fast" in path and "ultra-fast" not in path]
            if filtered_paths:
                mask_filepath=filtered_paths[0]
        else: 
            mask_filepath = mask_filepath[0] # glob returns a list. this gets first element of list
        print (f"timepoint: {timepoint} \nmask filepath: {mask_filepath}")
    else:
        print("No file found for patient_id", patient_id, "timepoint", timepoint)
    


        # Load nifti file as img. img has attributes 
    if img_filepath:
        print('Loading image nifti...')
        img, save_dir = load_nifti(img_filepath)
    elif mask_filepath and not img_filepath:
        print('Loading available mask as img...')
        img, save_dir=load_nifti(mask_filepath)
    else:
        print(f"neither img nor mask found for {patient_id} {timepoint}. Exiting.")
        sys.exit(1)
    
    # Load nifti file as img type - img has attributes 
    if mask_filepath:
        print('Loading mask nifti...')
        mask, save_dir = load_nifti(mask_filepath)
    else:    
        print(f"mask not found for {patient_id} {timepoint}. Exiting.")
        sys.exit(1)


    print(f"Starting contour extraction for {patient_id} {timepoint}...")
    # Extract voxel indices from patient_info csv
    #print(patient_info.columns)
    z_coord = patient_info.loc[(patient_info['patient_id'] == patient_id) & (patient_info['timepoint'] == timepoint), 'z_coord_slice'].values[0]
    # craniectomy skull end points
    antx = patient_info.loc[(patient_info['patient_id'] == patient_id) & (patient_info['timepoint'] == timepoint), 'anterior_x_coord'].values[0]
    anty = patient_info.loc[(patient_info['patient_id'] == patient_id) & (patient_info['timepoint'] == timepoint), 'anterior_y_coord'].values[0]
    postx = patient_info.loc[(patient_info['patient_id'] == patient_id) & (patient_info['timepoint'] == timepoint), 'posterior_x_coord'].values[0]
    posty = patient_info.loc[(patient_info['patient_id'] == patient_id) & (patient_info['timepoint'] == timepoint), 'posterior_y_coord'].values[0]
    side = patient_info.loc[(patient_info['patient_id'] == patient_id) & (patient_info['timepoint'] == timepoint), 'side_LR'].values[0]
    # corresponding baseline skull points
    bantx = patient_info.loc[(patient_info['patient_id'] == patient_id) & (patient_info['timepoint'] == timepoint), 'baseline_anterior_x_coord'].values[0]
    #banty = patient_info.loc[(patient_info['patient_id'] == patient_id) & (patient_info['timepoint'] == timepoint), 'baseline_anterior_y_coord'].values[0]
    bpostx = patient_info.loc[(patient_info['patient_id'] == patient_id) & (patient_info['timepoint'] == timepoint), 'baseline_posterior_x_coord'].values[0]
    #bposty = patient_info.loc[(patient_info['patient_id'] == patient_id) & (patient_info['timepoint'] == timepoint), 'baseline_posterior_y_coord'].values[0]

    #print("bpostx type: ",type(bpostx))
    #print("postx type: ",type(postx))
    """
    print(f"z coord slice index: {z_coord}")
    print(f"anterior x coord: {antx}")
    print(f"anterior y coord: {anty}")
    print(f"posterior x coord: {postx}")
    print(f"posterior y coord: {posty}")
    print(f"craniectomy side: {side}")
    """
    norm_img_slice, slice_img = extract_and_display_slice(img, save_dir, patient_id, timepoint, z_coord, disp_flag='n')
    norm_mask_slice, slice_mask = extract_and_display_slice(mask, save_dir, patient_id, timepoint, z_coord, disp_flag='n')

      
    print(f"z coord slice index: {z_coord}")
    print(f"anterior x coord: {antx}")
    print(f"anterior y coord: {anty}")
    print(f"posterior x coord: {postx}")
    print(f"posterior y coord: {posty}")
    print(f"craniectomy side: {side}")

    print(f"baseline anterior x coord: {bantx}")
    #print(f"baseline anterior y coord: {banty}")
    print(f"baseline posterior x coord: {bpostx}")
    #print(f"baseline posterior y coord: {bposty}")

    """
    plt.imshow(slice_img, cmap='gray',origin='lower' )
    plt.scatter(antx, anty, color='r')
    plt.scatter(postx, posty, color='r')
    plt.scatter(bantx, banty, color='b')
    plt.scatter(bpostx, bposty, color='b')
    plt.show()
    """

    # STEP 2: EXTRACT CONTOURS
    # Use mask: extract deformed side
    print("*****timepoint is:",timepoint)
    deformed_contour_x, deformed_contour_y = auto_boundary_detect(patient_id, timepoint, norm_mask_slice, antx, anty, postx, posty, side)
    if np.all(np.array(deformed_contour_x) == None):
        continue

    # create input for side to do other side
    flipside = flipside_func(side)
    
    # Use mask: extract baseline side
    baseline_contour_x, baseline_contour_y = auto_boundary_detect(patient_id, timepoint, norm_mask_slice, antx, anty, postx, posty, flipside)
    #print("baseline_contour_x type: ", type(baseline_contour_x))

    # get mirror line from baseline vs deformed contours in x 
    skull_end_x = np.array([antx, postx], dtype=int)
    skull_end_y = np.array([anty, posty], dtype=int)
    baseline_skull_end_x = np.array([bantx, bpostx], dtype=int)

    #print("skull_end_x type: ", type(skull_end_x))
    #print("skull_end_y type: ", type(skull_end_y))
    #print("baseline_skull_end_x type: ", type(baseline_skull_end_x))
    
    m, c, Y = get_mirror_line(skull_end_y, skull_end_x, baseline_skull_end_x)
    #m, c, Y = get_mirror_line(deformed_contour_y, deformed_contour_x, baseline_contour_x)


    # create reflected contours across line
    reflected_contour_x, reflected_contour_y = reflect_across_line(m, c, baseline_contour_x, baseline_contour_y)


    # STEP 3: PLOT CONTOURS AND MIDLINE
    # account for if gradient = 0
    if m == 0:
        x_values = [c for y in baseline_contour_y]
    else:
        x_values = [(y - c) / m for y in baseline_contour_y]
    # Create a DataFrame to store the y and corresponding x values
    line_data = pd.DataFrame({
        'y': baseline_contour_y,
        'x': x_values
    })

    plt.imshow(norm_img_slice, cmap='gray')
    ## Adjust the y-axis to display in the original image's orientation
    plt.gca().invert_yaxis()
    plt.plot(line_data['x'], line_data['y'], label=f'Line: y = {m}x + {c}', color='white', lw=0.5, linestyle='dashed')
    plt.scatter(deformed_contour_x, deformed_contour_y, s=2, color='red')
    plt.scatter(skull_end_x, skull_end_y, s=10, color='magenta')
    plt.scatter(baseline_skull_end_x, skull_end_y, s=10, color='magenta')
    plt.scatter(baseline_contour_x, baseline_contour_y, s=2, color='cyan')
    plt.scatter(reflected_contour_x, reflected_contour_y, s=2, color='green')
    plt.title(f"{patient_id} {timepoint}")
    filename = save_dir +"/" + timepoint+".png"
    plt.savefig(filename)
    #print("file path: ",filename)
    plt.show()


    print(f"Image {timepoint}.png saved to {save_dir}")
    print(f"Contour point extraction for {patient_id} {timepoint} complete. \n")

print("Plots completed for all specified timepoints.")
print("specify run only particular patient id by doing \n python contour_plot_main.py <patient_id>")

