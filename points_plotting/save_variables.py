import numpy as np
import os

def save_arrays_to_directory(directory, filename, **arrays):
    """
    Save multiple NumPy arrays to a specified directory.

    Parameters:
    - directory: The directory where the file will be saved.
    - filename: The name of the file to save the arrays to.
    - arrays: Keyword arguments where keys are the variable names to use when loading the arrays back,
              and values are the NumPy arrays to be saved.
    """
    # Check if the directory exists; create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Construct the full file path
    filepath = os.path.join(directory, filename)
    
    # Save the arrays to the specified file
    np.savez(filepath, **arrays)
    
    return 0
