# THIS FUNCTION LOADS A NIFTI FILE AND RETURNS IT 

import os
import nibabel as nib


def load_nifti(nifti_file_path):
    # Extract the directory from nifti_file_path
    nifti_directory = os.path.dirname(nifti_file_path)

    # Define the directory for saving the image
    save_directory = os.path.join(nifti_directory, 'python_plots')
    
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
