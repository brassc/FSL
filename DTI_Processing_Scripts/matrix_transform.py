#!/usr/bin/env python3
"""
Script to transform coordinates from T1 space to DTI space using transformation matrices
"""

import os
import sys
import csv
import numpy as np
from pathlib import Path

# Define the base path
BASEPATH = "/home/cmb247/rds/hpc-work/April2025_DWI/"
REPO_LOCATION = os.getcwd()  # Use current working directory

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
    
    # Return the transformed coordinates
    return new_coord[0], new_coord[1], new_coord[2]

def main():
    # Create a copy of the original CSV
    input_csv = os.path.join(REPO_LOCATION, "DTI_Processing_Scripts", "LEGACY_coords.sh")
    output_csv = os.path.join(REPO_LOCATION, "DTI_Processing_Scripts", "LEGACY_DTI_coords_transformed.csv")
    temp_csv = os.path.join(REPO_LOCATION, "DTI_Processing_Scripts", "LEGACY_DTI_coords_temp.csv")
    
    # Copy the original file
    os.system(f"cp '{input_csv}' '{os.path.join(REPO_LOCATION, 'DTI_Processing_Scripts', 'LEGACY_DTI_coords.csv')}'")
    
    # Read the input CSV file
    all_rows = []
    with open(input_csv, 'r', newline='') as csvfile:
        # Read the header
        header = csvfile.readline().strip().split(',')
        header = [col.strip() for col in header]
        
        # Create the new header with added columns
        new_header = header[:]
        # Insert new columns before COMMENTS
        comments_index = header.index("COMMENTS") if "COMMENTS" in header else len(header)
        new_columns = [
            "dwispace anterior x", "dwispace anterior y", "dwispace anterior z",
            "dwispace posterior x", "dwispace posterior y", "dwispace posterior z",
            "dwispace baseline anterior x", "dwispace baseline anterior y", "dwispace baseline anterior z",
            "dwispace baseline posterior x", "dwispace baseline posterior y", "dwispace baseline posterior z"
        ]
        for i, col in enumerate(new_columns):
            new_header.insert(comments_index + i, col)
        
        # Read the rest of the rows
        reader = csv.reader(csvfile)
        for row in reader:
            row = [col.strip() for col in row]
            all_rows.append(row)
    
    # Process each row and apply transformations
    transformed_rows = []
    for row in all_rows:
        # Skip rows with insufficient columns
        if len(row) < len(header):
            print(f"Warning: Skipping row with insufficient columns: {row}")
            continue
            
        # Extract data from the row
        row_dict = dict(zip(header, row))
        excluded = row_dict.get("excluded?", "").strip()
        patient_id = row_dict.get("patient ID", "").strip()
        timepoint = row_dict.get("timepoint", "").strip()
        z_coord = row_dict.get("z coord (slice)", "").strip()
        anterior_x = row_dict.get("anterior x coord", "").strip()
        anterior_y = row_dict.get("anterior y coord", "").strip()
        posterior_x = row_dict.get("posterior x coord", "").strip()
        posterior_y = row_dict.get("posterior y coord", "").strip()
        side = row_dict.get("side (L/R)", "").strip()
        baseline_anterior_x = row_dict.get("baseline anterior x coord", "").strip()
        baseline_posterior_x = row_dict.get("baseline posterior x coord", "").strip()
        comments = row_dict.get("COMMENTS", "").strip()
        
        # Initialize transformed coordinates as empty
        dwispace_coords = [""] * 12  # 12 new coordinate fields
        
        # Only process non-excluded entries (excluded = 0)
        if excluded == "0":
            # Define transformation matrix path
            transform_matrix = os.path.join(BASEPATH, patient_id, timepoint, "Stefan_preprocessed_DWI_space", "T1_to_DTI.mat")
            
            if os.path.isfile(transform_matrix):
                print(f"Processing patient {patient_id} at timepoint {timepoint}...")
                
                try:
                    # Transform the coordinates
                    anterior_dwi = apply_transform(anterior_x, anterior_y, z_coord, transform_matrix)
                    posterior_dwi = apply_transform(posterior_x, posterior_y, z_coord, transform_matrix)
                    baseline_anterior_dwi = apply_transform(baseline_anterior_x, anterior_y, z_coord, transform_matrix)
                    baseline_posterior_dwi = apply_transform(baseline_posterior_x, posterior_y, z_coord, transform_matrix)
                    
                    # Store the transformed coordinates
                    dwispace_coords = [
                        str(round(anterior_dwi[0], 3)), str(round(anterior_dwi[1], 3)), str(round(anterior_dwi[2], 3)),
                        str(round(posterior_dwi[0], 3)), str(round(posterior_dwi[1], 3)), str(round(posterior_dwi[2], 3)),
                        str(round(baseline_anterior_dwi[0], 3)), str(round(baseline_anterior_dwi[1], 3)), str(round(baseline_anterior_dwi[2], 3)),
                        str(round(baseline_posterior_dwi[0], 3)), str(round(baseline_posterior_dwi[1], 3)), str(round(baseline_posterior_dwi[2], 3))
                    ]
                except Exception as e:
                    print(f"Error processing coordinates for patient {patient_id} at timepoint {timepoint}: {e}")
            else:
                print(f"Warning: Transform matrix not found for patient {patient_id} at timepoint {timepoint}")
        
        # Create a new row with the transformed coordinates
        new_row = row[:]
        # Insert the transformed coordinates before the comments column
        comments_index = header.index("COMMENTS") if "COMMENTS" in header else len(row)
        for i, coord in enumerate(dwispace_coords):
            new_row.insert(comments_index + i, coord)
        
        transformed_rows.append(new_row)
    
    # Write the transformed data to the output CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(new_header)
        writer.writerows(transformed_rows)
    
    print(f"Coordinate transformation complete. Results saved to {output_csv}")

if __name__ == "__main__":
    main()