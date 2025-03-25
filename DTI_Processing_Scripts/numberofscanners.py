import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Load Sophie's scan database
scandataloc = 'Sophie_Data/Sophies_scan_database_20220822.csv'
scandata = pd.read_csv(scandataloc)
print(scandata.columns)

# get patient data
batch1=pd.read_csv('Image_Processing_Scripts/included_patient_info.csv')
batch2=pd.read_csv('Image_Processing_Scripts/batch2_included_patient_info.csv')

#Clean cols
batch1.columns = batch1.columns.str.strip().str.replace(' ', '_').str.replace('?', '').str.replace('(', '').str.replace(')', '').str.lower()
batch2.columns = batch2.columns.str.strip().str.replace(' ', '_').str.replace('?', '').str.replace('(', '').str.replace(')', '').str.lower()
batch1 = batch1.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
batch2 = batch2.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
batch1['patient_id'] = batch1['patient_id'].astype(str)


# concatenate the two batches
batch = pd.concat([batch1, batch2])
# pop the row if excluded == 1
batch = batch[batch['excluded'] != 1]
print(batch)


# get the patient ids
patientids = batch['patient_id'].unique()
print(patientids)


# get scanner data for patients

all_scanner_data = pd.DataFrame(columns=['patient_id', 'Cohort', 'Site', 'Model', 'Days_since_injury', 'Scan_date'])
for patient in patientids:
    # if patient is a 5 digit number, look in WBIC_ID column
    if len(str(patient)) == 5:
        patient_data=scandata[scandata['WBIC_ID'].astype(str).str.contains(str(patient))].copy()
        # get only 'Cohort', 'Site', 'Model', 'Days_since_injury', 'Scan_date' columns
        patient_data['WBIC_ID'] = patient_data['WBIC_ID'].astype(str).str.split('.').str[0]
        patient_data = patient_data[['WBIC_ID','Cohort', 'Site', 'Model', 'Days_since_injury', 'Scan_date']]
        # rename WBIC_ID to patient_id
        patient_data.rename(columns={'WBIC_ID':'patient_id'}, inplace=True)
        # append the data to df with all patients
        all_scanner_data = pd.concat([all_scanner_data, patient_data], ignore_index=True)
    elif len(str(patient)) == 7:
        print("GUPI SCANNING")
        patient_data=scandata[scandata['GUPI'].astype(str).str.contains(str(patient))].copy()
        patient_data = patient_data[['GUPI','Cohort', 'Site', 'Model', 'Days_since_injury', 'Scan_date']]
        patient_data.rename(columns={'GUPI':'patient_id'}, inplace=True)
        all_scanner_data = pd.concat([all_scanner_data, patient_data], ignore_index=True)

# get the number of scanners
scanners = all_scanner_data['Model'].unique()
print(scanners)
print(len(scanners))
# Print rows where Model is NaN
if all_scanner_data[all_scanner_data['Model'].isna()].shape[0] > 0:
    print(all_scanner_data[all_scanner_data['Model'].isna()][['patient_id', 'Model']])

sites=all_scanner_data['Site'].unique()
print(sites)
print(len(sites))

print(all_scanner_data.head)
# save .csv file
all_scanner_data.to_csv('DTI_Processing_Scripts/patient_scanner_data.csv', index=False)

# find ranges
timepoint_data=pd.read_csv('DTI_Processing_Scripts/patient_scanner_data_with_timepoints.csv')
print(timepoint_data.columns)
timepoint_data.columns = timepoint_data.columns.str.strip().str.replace(' ', '_').str.replace('?', '').str.replace('(', '').str.replace(')', '').str.lower()


# Define order
order = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']

# Get unique timepoints (correctly)
unique_timepoints = timepoint_data['timepoint'].unique()

ranges = []
for timepoint in unique_timepoints:
    # Skip NaN timepoints
    if isinstance(timepoint, float) and np.isnan(timepoint):
        continue
        
    # Set lower bound for ultra-fast to 0
    if timepoint.strip() == 'ultra-fast':
        min_days_since_injury = 0
    else:
        min_days_since_injury = timepoint_data[timepoint_data['timepoint'] == timepoint]['days_since_injury'].min()
        
    # Find maximum associated with timepoint
    max_days_since_injury = timepoint_data[timepoint_data['timepoint'] == timepoint]['days_since_injury'].max()
    
    print(f'{timepoint}: {min_days_since_injury} - {max_days_since_injury}')
    
    # Add to list
    ranges.append([timepoint.strip(), min_days_since_injury, max_days_since_injury])

# Create DataFrame
df = pd.DataFrame(ranges, columns=['timepoint', 'lower_bound', 'upper_bound'])

# Reorder DataFrame based on predefined order
df['order'] = df['timepoint'].apply(lambda x: order.index(x) if x in order else 999)
df = df.sort_values('order').drop('order', axis=1).reset_index(drop=True)

print(df)

# First, convert days to hours
df['lower_bound_hours'] = df['lower_bound'] * 24
df['upper_bound_hours'] = df['upper_bound'] * 24

print("Original ranges in hours:")
print(df[['timepoint', 'lower_bound_hours', 'upper_bound_hours']])

# Now make ranges continuous
continuous_lower = []
continuous_upper = []

for i in range(len(df)):
    if i == 0:
        # First timepoint starts at its original lower bound (usually 0)
        continuous_lower.append(df.iloc[i]['lower_bound_hours'])
    else:
        # Other timepoints start exactly where the previous one ended
        continuous_lower.append(continuous_upper[i-1])
    
    if i == len(df) - 1:
        # Last timepoint keeps its original upper bound
        continuous_upper.append(df.iloc[i]['upper_bound_hours'])
    else:
        # For other timepoints, set upper bound exactly to next lower bound
        continuous_upper.append(df.iloc[i+1]['lower_bound_hours'])

# Update DataFrame with continuous ranges
df['continuous_lower_hours'] = continuous_lower
df['continuous_upper_hours'] = continuous_upper

# Round for cleaner values
df['continuous_lower_hours'] = df['continuous_lower_hours'].round(1)
df['continuous_upper_hours'] = df['continuous_upper_hours'].round(1)

print("\nContinuous ranges in hours:")
print(df[['timepoint', 'continuous_lower_hours', 'continuous_upper_hours']])
