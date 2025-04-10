#!/usr/bin/env python3
"""
Script to transform coordinates from T1 space to DTI space using transformation matrices
"""

import os
import sys
import csv
import logging
import numpy as np
from pathlib import Path
import pandas as pd



# Define the base path
BASEPATH = "/home/cmb247/rds/hpc-work/April2025_DWI/"
REPO_LOCATION = os.getcwd()  # Use current working directory
# Set up logging to both console and log file
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Output to console
                        logging.FileHandler(f"{REPO_LOCATION}/DTI_Processing_Scripts/coordinate_transformation.log")  # Output to log file
                    ])

def apply_transform(x, y, z, matrix_file):
    """
    Apply transformation matrix to coordinates
    
    Args:
        x, y, z: Input coordinates
        matrix_file: Path to the transformation matrix file
        
    Returns:
        tuple: Transformed coordinates (x, y, z)
    """
    # Create the coordinate vector (homogeneous coordinates)
    coord = np.array([float(x), float(y), float(z), 1.0])
    
    # Read the transformation matrix
    matrix = np.loadtxt(matrix_file)
    
    # Apply the transformation
    new_coord = np.dot(matrix, coord)

    # Adjust for the difference in voxel size (DWI has voxel size of 2, T1 has voxel size of 1)
    adjusted_coord = new_coord[:3] * 0.5  # Scaling by 1/2
    
    # Return the adjusted transformed coordinates
    return adjusted_coord[0], adjusted_coord[1], adjusted_coord[2]
    

def main():
    # Create a copy of the original CSV
    input_csv = os.path.join(REPO_LOCATION, "DTI_Processing_Scripts", "LEGACY_coords.csv")
    output_csv = os.path.join(REPO_LOCATION, "DTI_Processing_Scripts", "LEGACY_DTI_coords_transformed.csv")
    temp_csv = os.path.join(REPO_LOCATION, "DTI_Processing_Scripts", "LEGACY_DTI_coords_temp.csv")


    # print(f"Copying {input_csv} to {temp_csv}...")
    
    
    # Copy the original file
    os.system(f"cp '{input_csv}' '{os.path.join(REPO_LOCATION, 'DTI_Processing_Scripts', 'LEGACY_DTI_coords.csv')}'")
    
    
    # Read the input CSV file
    all_rows = []
    with open(input_csv, 'r', newline='') as csvfile:
        # Read the header
        header = [col.strip().replace(" ", "_").lower() for col in csvfile.readline().strip().split(',')]
        header = [col.strip() for col in header]
        # print(f"Header: {header}")
        
        
        # Create the new header with added columns
        new_header = header[:]
        # Insert new columns before COMMENTS
        comments_index = header.index("comments") if "comments" in header else len(header)
        new_columns = [
            "dwispace_anterior_x", "dwispace_anterior_y", "dwispace_anterior_z",
            "dwispace_posterior_x", "dwispace_posterior_y", "dwispace_posterior_z",
            "dwispace_baseline_anterior_x", "dwispace_baseline_anterior_y", "dwispace_baseline_anterior_z",
            "dwispace_baseline_posterior_x", "dwispace_baseline_posterior_y", "dwispace_baseline_posterior_z"
        ]
        # print(f"New columns: {new_columns}")
        
        for i, col in enumerate(new_columns):
            new_header.insert(comments_index + i, col)
        
        print(f"New header: {new_header}")
        
        # Read the rest of the rows
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:  # Skip empty rows
                continue
            row = [col.strip() for col in row]
            all_rows.append(row)

        # print rows with both letter and number in patient_id
        for row in all_rows:
            patient_id = row[header.index("patient_id")].strip()
            if any(c.isdigit() for c in patient_id) and any(c.isalpha() for c in patient_id):
                print(f"Row with mixed patient_id: {row}")
        



    




    # Process each row and apply transformations
    transformed_rows = []

    for row in all_rows:
        # Skip rows with insufficient columns
        if len(row) < len(header):
            #print(f"Warning: Row too short, padding with empty strings: {row}")
            row += [''] * (len(header) - len(row))
            #print(f"Row after padding: {row}")
            
          
        # Extract data from the row
        row_dict = dict(zip(header, row))

        # # Optional: Print the mapped dictionary for inspection
        # print("\nMapped Row Dictionary:")
        # for key in header:
        #     print(f"{key}: {row_dict.get(key, '')}")

        
        # Accessing and stripping values
        excluded = row_dict.get("excluded?", "").strip()
        patient_id = row_dict.get("patient_id", "").strip()
        timepoint = row_dict.get("timepoint", "").strip()
        z_coord = row_dict.get("z_coord_(slice)", "").strip()
        anterior_x = row_dict.get("anterior_x_coord", "").strip()
        anterior_y = row_dict.get("anterior_y_coord", "").strip()
        posterior_x = row_dict.get("posterior_x_coord", "").strip()
        posterior_y = row_dict.get("posterior_y_coord", "").strip()
        side = row_dict.get("side_(l/r)", "").strip()
        baseline_anterior_x = row_dict.get("baseline_anterior_x_coord", "").strip()
        baseline_posterior_x = row_dict.get("baseline_posterior_x_coord", "").strip()
        comments = row_dict.get("comments", "").strip()
        
        # Initialize transformed coordinates as empty
        dwispace_coords = [""] * 12  # 12 new coordinate fields
        
        
        # Only process non-excluded entries (excluded = 0)
        if excluded == "0":
            # Check if patient_id contains both numbers and letters
            has_mixed_id = any(c.isdigit() for c in patient_id) and any(c.isalpha() for c in patient_id)
            
            # If patient_id has mixed characters, handle the special case
            # # skip numeric patient_id
            # if not has_mixed_id:
            #     # skip
            #     print(f"Skipping patient_id {patient_id} as it does not contain both letters and numbers.")
            #     continue

            if has_mixed_id:
                print(f"timepoint: {timepoint}")
                
                # Check if timepoint is numeric (5 digits)
                if timepoint.isdigit() and len(timepoint) == 5:
                    
                    # Find the matching timepoint directory
                    
                    patient_path = os.path.join(BASEPATH, patient_id)
                    matching_dirs = []
                    
                    # List directories in the patient folder
                    if os.path.exists(patient_path):
                        
                        for item in os.listdir(patient_path):
                            item_path = os.path.join(patient_path, item)
                            # Check if it's a directory and matches the expected pattern
                            if os.path.isdir(item_path) and item.startswith(f"Hour-{timepoint}") and "_dwi" in item:
                                matching_dirs.append(item)
                                
                    
                    if matching_dirs:
                        # Use the first matching directory (or you could add more logic if multiple matches)
                        timepoint_dir = matching_dirs[0]
                        print(f"timepoint_dir: {timepoint_dir}")
                        
                        # Use the alternative path structure for transform matrix
                        transform_matrix = os.path.join(BASEPATH, patient_id, timepoint_dir, 
                                                    "T1_space_bet/coordinates_files/", "T1_to_DTI.mat")
            else:
                # Standard case - use the original path structure
                transform_matrix = os.path.join(BASEPATH, patient_id, timepoint, 
                                            "Stefan_preprocessed_DWI_space", "T1_to_DTI.mat")
            
            print(f"Transform matrix path: {transform_matrix}")


            if os.path.isfile(transform_matrix):
                print(f"Processing patient {patient_id} at timepoint {timepoint}...")
                
                
                
                try:
                    # Transform the coordinates
                    anterior_dwi = apply_transform(anterior_x, anterior_y, z_coord, transform_matrix)
                    posterior_dwi = apply_transform(posterior_x, posterior_y, z_coord, transform_matrix)
                    baseline_anterior_dwi = apply_transform(baseline_anterior_x, anterior_y, z_coord, transform_matrix)
                    baseline_posterior_dwi = apply_transform(baseline_posterior_x, posterior_y, z_coord, transform_matrix)

                    # Convert the transformed coordinates to numpy arrays (mutable) if they aren't already
                    anterior_dwi = np.array(anterior_dwi)
                    posterior_dwi = np.array(posterior_dwi)
                    baseline_anterior_dwi = np.array(baseline_anterior_dwi)
                    baseline_posterior_dwi = np.array(baseline_posterior_dwi)
                    
                    # # Print transformed coordinates as integers
                    # print(f"Transformed coordinates for patient {patient_id} at timepoint {timepoint}:")
                    # print(f"Anterior DWI: ({int(round(anterior_dwi[0]))}, {int(round(anterior_dwi[1]))}, {int(round(anterior_dwi[2]))})")
                    # print(f"Posterior DWI: ({int(round(posterior_dwi[0]))}, {int(round(posterior_dwi[1]))}, {int(round(posterior_dwi[2]))})")
                    # print(f"Baseline Anterior DWI: ({int(round(baseline_anterior_dwi[0]))}, {int(round(baseline_anterior_dwi[1]))}, {int(round(baseline_anterior_dwi[2]))})")
                    # print(f"Baseline Posterior DWI: ({int(round(baseline_posterior_dwi[0]))}, {int(round(baseline_posterior_dwi[1]))}, {int(round(baseline_posterior_dwi[2]))})")

                    # Log transformed coordinates as integers
                    logging.info(f"Transformed coordinates for patient {patient_id} at timepoint {timepoint}:")
                    logging.info(f"Anterior DWI: ({int(round(anterior_dwi[0]))}, {int(round(anterior_dwi[1]))}, {int(round(anterior_dwi[2]))})")
                    logging.info(f"Posterior DWI: ({int(round(posterior_dwi[0]))}, {int(round(posterior_dwi[1]))}, {int(round(posterior_dwi[2]))})")
                    logging.info(f"Baseline Anterior DWI: ({int(round(baseline_anterior_dwi[0]))}, {int(round(baseline_anterior_dwi[1]))}, {int(round(baseline_anterior_dwi[2]))})")
                    logging.info(f"Baseline Posterior DWI: ({int(round(baseline_posterior_dwi[0]))}, {int(round(baseline_posterior_dwi[1]))}, {int(round(baseline_posterior_dwi[2]))})")

                    # if difference in z coordinate between any of the transformed coordinates is greater than 10, print a warning and 
                    # take the average of the z coordinates and set that as z for all the transformed coordinates
                    z_coords = [anterior_dwi[2], posterior_dwi[2], baseline_anterior_dwi[2], baseline_posterior_dwi[2]]
                    if max(z_coords) - min(z_coords) > 2:
                        # print(f"Warning: Large difference in z coordinates for patient {patient_id} at timepoint {timepoint}. Taking average.")
                        logging.warning(f"Large difference in z coordinates for patient {patient_id} at timepoint {timepoint}. Taking average.")
                        avg_z = int(round(np.mean(z_coords)))
                        anterior_dwi[2] = avg_z
                        posterior_dwi[2] = avg_z
                        baseline_anterior_dwi[2] = avg_z
                        baseline_posterior_dwi[2] = avg_z
                        # print(f"New z coordinate: {avg_z}")
                        logging.info(f"New z coordinate: {avg_z}")
                    
                    y_coords = [anterior_dwi[1], posterior_dwi[1], baseline_anterior_dwi[1], baseline_posterior_dwi[1]]
                    if anterior_dwi[1] != baseline_anterior_dwi[1]:
                        # Find average anterior y coordinate out of anterior_dwi[1] and baseline_anterior_dwi[1]
                        avg_ant_y = int(round(np.mean([anterior_dwi[1], baseline_anterior_dwi[1]])))
                        # set the y coordinate of the anterior to the average
                        anterior_dwi[1] = avg_ant_y
                        # set the y coordinate of the baseline anterior to the average
                        baseline_anterior_dwi[1] = avg_ant_y
                        logging.warning(f"Setting y coordinate to average of anterior and baseline anterior for patient {patient_id} at timepoint {timepoint}.")


                        # set the y coordinate of the anterior to the y coordinate of the baseline anterior
                        #baseline_anterior_dwi[1] = anterior_dwi[1]
                        # print(f"Warning: y coordinate of anterior and baseline anterior do not match for patient {patient_id} at timepoint {timepoint}. Setting baseline anterior to anterior.")
                        #logging.warning(f"y coordinate of anterior and baseline anterior do not match for patient {patient_id} at timepoint {timepoint}. Setting baseline anterior y equal to anterior y ({anterior_dwi[1]}).")
                    
                    if posterior_dwi[1] != baseline_posterior_dwi[1]:
                        # Find average posterior y coordinate out of posterior_dwi[1] and baseline_posterior_dwi[1]
                        avg_ant_y = int(round(np.mean([posterior_dwi[1], baseline_posterior_dwi[1]])))
                        # set the y coordinate of the posterior to the average
                        posterior_dwi[1] = avg_ant_y
                        # set the y coordinate of the baseline posterior to the average
                        baseline_posterior_dwi[1] = avg_ant_y
                        logging.warning(f"Setting y coordinate to average of posterior and baseline posterior for patient {patient_id} at timepoint {timepoint}.")

                        # # set the y coordinate of the posterior to the y coordinate of the baseline posterior
                        # baseline_posterior_dwi[1] = posterior_dwi[1]
                        # # print(f"Warning: y coordinate of posterior and baseline posterior do not match for patient {patient_id} at timepoint {timepoint}. Setting baseline posterior to posterior.")
                        # logging.warning(f"y coordinate of posterior and baseline posterior do not match for patient {patient_id} at timepoint {timepoint}. Setting basline posterior y equal to posterior y ({posterior_dwi[1]}).")


                    # Store the transformed coordinates
                    dwispace_coords = [
                        str(int(round(anterior_dwi[0]))), str(int(round(anterior_dwi[1]))), str(int(round(anterior_dwi[2]))),
                        str(int(round(posterior_dwi[0]))), str(int(round(posterior_dwi[1]))), str(int(round(posterior_dwi[2]))),
                        str(int(round(baseline_anterior_dwi[0]))), str(int(round(baseline_anterior_dwi[1]))), str(int(round(baseline_anterior_dwi[2]))),
                        str(int(round(baseline_posterior_dwi[0]))), str(int(round(baseline_posterior_dwi[1]))), str(int(round(baseline_posterior_dwi[2])))
                    ]
                    
                   
                except Exception as e:
                    print(f"Error processing coordinates for patient {patient_id} at timepoint {timepoint}: {e}")
            else:
                # print(f"Warning: Transform matrix not found for patient {patient_id} at timepoint {timepoint}")
                logging.warning(f"Transform matrix not found for patient {patient_id} at timepoint {timepoint}")

        print(f"writing this print output log file...")
                # Construct the new row with original data and transformed coordinates
        new_row = row.copy()  # Start with the original row
        
        # Insert the transformed coordinates before the comments column
        comments_index = header.index("comments") if "comments" in header else len(row)
        
        # Add extra elements to the row if needed to ensure we have enough space
        while len(new_row) < comments_index:
            new_row.append("")
            
        # Insert transformed coordinates at the right position
        for i, coord in enumerate(dwispace_coords):
            if comments_index + i >= len(new_row):
                new_row.append(coord)  # Append if we're beyond the current row length
            else:
                new_row.insert(comments_index + i, coord)  # Insert otherwise
                
        # Ensure the row has the right length (should match new_header)
        while len(new_row) < len(new_header):
            new_row.append("")
            
        # Trim if too long
        if len(new_row) > len(new_header):
            new_row = new_row[:len(new_header)]
            
        transformed_rows.append(new_row)

        
    # Write the transformed data to the output CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(new_header)
        writer.writerows(transformed_rows)
    
    print(f"Coordinate transformation complete. Results saved to {output_csv}")

if __name__ == "__main__":
    main()