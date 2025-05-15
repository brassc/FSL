import pandas as pd
import numpy as np
import os
import sys
import ast
import re
# Load input CSV file
input_file='DTI_Processing_Scripts/results/all_metrics_5x4vox_NEW_filtered_all_values.csv'
output_file='DTI_Processing_Scripts/results/all_metrics_5x4vox_NEW_filtered_median.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file)
result_df = pd.DataFrame()

# copy patient_id and timepoint columns to the result_df
result_df['patient_id'] = df['patient_id']
result_df['timepoint'] = df['timepoint']

# Process each column in the dataframe
for column in df.columns:
    # Skip patient_id and timepoint columns
    if column in ['patient_id', 'timepoint']:
        continue

    # Process the column
    result_column = []
    
    for value in df[column]:
        try:
            # Convert string representation of array to actual array
            # print(f"Processing column: {column}, value: {value}")
            # Extract all floating point numbers from the string
            if isinstance(value, str):
                floats_array = [float(x) for x in re.findall(r'[-+]?\d*\.\d+|\d+', value)]
            else:
                floats_array = [float(value)] if not pd.isna(value) else []
            
            # print(f"Extracted floats: {floats_array}")
            # Convert to numpy array
            array = np.array(floats_array)
            # print(f"Converted to numpy array: {array}")
            #sys.exit()
            
            # Apply filtering for FA columns only
            if column.startswith('FA_'):
                filtered_array = [x for x in array if 0.05 <= x <= 0.8]
                # If filtering removes all values, use original array
                if not filtered_array:
                    filtered_array = array
            else:  # For MD columns
                filtered_array = array
            
            # print(f"Filtered array for {column}: {filtered_array}")
            # sys.exit()
            # Calculate median
            if len(filtered_array) > 0:
                median_value = np.median(filtered_array)
            else:
                median_value = np.nan
            
            result_column.append(median_value)

            # 
        except (ValueError, SyntaxError, TypeError):
            # If value isnt an array, keep it as it is
            result_column.append(value)
    
    # Add processed column to result df, using the same column name
    result_df[column] = result_column
    # print (f"Processed column: {column}, result: {result_column}")

result_df.to_csv(output_file, index=False, float_format='%.10f')

print(f"Processing complete. Output saved to {output_file}")





