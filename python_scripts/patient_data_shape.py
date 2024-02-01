import numpy as np
import pandas as pd


# Read data in 
data=pd.read_csv('/Users/charlottebrass/repos/FSL/patient_timeline_map.csv')

# Fill empty cells with NaN
data = data.fillna(0)

# Drop the last column from the data
trimmed_data = data.iloc[:, :-1]

# Convert to int
for col in trimmed_data.columns[:8]:  # Selecting the first 8 columns
    trimmed_data[col] = trimmed_data[col].astype(int)

# Create a copy of the trimmed data
df = trimmed_data.copy()

# How many scans per patient
df['No. of Scans'] = df.iloc[:, -7:].sum(axis=1)

# Split into bifrontal and hemi dataframes
bifrontal_df = df.iloc[-7:, :]
hemi_df = df.iloc[:25, :]

# Filter hemi dataset for patients with more than one scan
hemi_df_filtered = hemi_df.loc[hemi_df['No. of Scans'] > 1]
# Add these together
hemi_total_scans = hemi_df_filtered['No. of Scans'].sum()
total_number_of_hemipatients_w_2ormore_scans = len(hemi_df_filtered)

print(hemi_total_scans, total_number_of_hemipatients_w_2ormore_scans)

bifrontal_df_filtered = bifrontal_df.loc[bifrontal_df['No. of Scans'] > 1]
bif_total_scans = bifrontal_df_filtered['No. of Scans'].sum()
total_number_of_bifpatients_w_2ormore_scans = len(bifrontal_df_filtered)

print(bif_total_scans, total_number_of_bifpatients_w_2ormore_scans)











