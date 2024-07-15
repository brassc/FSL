import os

def ensure_directory_exists(patient_id, patient_timepoint):
    """
    Check if a directory exists, and if not, create it.

    Parameters:
    - directory_path: The path to the directory to check and possibly create.
    """

    directory_path=f"/home/cmb247/repos/FSL/points_plotting/points_dir/{patient_id}_{patient_timepoint}"
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory created: {directory_path}")
        else:
            print(f"Directory already exists: {directory_path}")
    except Exception as e:
        print(f"Failed to create directory: {directory_path}. Error: {e}")

    return directory_path
