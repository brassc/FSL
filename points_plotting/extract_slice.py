import nibabel as nib
import numpy as np
from PIL import Image

def extract_and_display_slice(nifti_file_path, scanner_coords):
    log_file = "nifti_load_log.txt"
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

    # Get the affine transformation matrix
    affine = img.affine

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

    # Rotate 90 degrees to expected orientation (i.e., rotate counter-clockwise)
    rotated_slice_image = slice_image.rotate(90, expand=True)

    # Save and display the image
    rotated_slice_image.save('extracted_slice.png')  # Save the corrected slice image
    rotated_slice_image.show()

