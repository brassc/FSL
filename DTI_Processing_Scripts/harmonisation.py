import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuroCombat import neuroCombat
import sys


def merge_scanner_info_with_metrics(metrics_df, scanner_info_df, output_filename):
    """
    Merge scanner information (Cohort, Site, Model) with metrics data.
    
    Args:
        metrics_df: DataFrame containing metrics data
        scanner_info_df: DataFrame containing scanner information
        output_filename: Path to save the merged data
        
    Returns:
        DataFrame: The merged data with scanner information added
    """
    # Clean data from spaces everywhere
    metrics_df.columns = metrics_df.columns.str.strip().str.replace(' ', '_').str.replace('?', '').str.replace('(', '').str.replace(')', '').str.lower()
    metrics_df = metrics_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    metrics_df['patient_id'] = metrics_df['patient_id'].astype(str)
    metrics_df['timepoint'] = metrics_df['timepoint'].astype(str)
    
    # Add new columns for scanner information
    metrics_df['Cohort'] = None
    metrics_df['Site'] = None
    metrics_df['Model'] = None
    
    # Create a mapping dictionary for 5-digit patient IDs
    mapping_dict = {}
    for idx, row in scanner_info_df[scanner_info_df['patient_id'].astype(str).str.len() == 5].iterrows():
        patient_id = str(row['patient_id'])
        timepoint = str(row['timepoint'])
        key = (patient_id, timepoint)
        mapping_dict[key] = {
            'Cohort': row['Cohort'],
            'Site': row['Site'],
            'Model': row['Model']
        }
    
    # Process 7-digit patient IDs directly
    seven_digit_patients = scanner_info_df[scanner_info_df['patient_id'].astype(str).str.len() == 7]['patient_id'].unique()
    for patient_id in seven_digit_patients:
        patient_rows = scanner_info_df[scanner_info_df['patient_id'] == patient_id]
        if not patient_rows.empty:
            first_row = patient_rows.iloc[0]
            cohort = first_row['Cohort']
            site = first_row['Site']
            model = first_row['Model']
            
            # Update all rows for this patient
            for idx, row in metrics_df[metrics_df['patient_id'] == patient_id].iterrows():
                metrics_df.at[idx, 'Cohort'] = cohort
                metrics_df.at[idx, 'Site'] = site
                metrics_df.at[idx, 'Model'] = model
    
    # Update 5-digit patient IDs using the mapping dictionary
    for idx, row in metrics_df[metrics_df['patient_id'].astype(str).str.len() == 5].iterrows():
        patient_id = str(row['patient_id'])
        timepoint = str(row['timepoint'])
        key = (patient_id, timepoint)
        if key in mapping_dict:
            metrics_df.at[idx, 'Cohort'] = mapping_dict[key]['Cohort']
            metrics_df.at[idx, 'Site'] = mapping_dict[key]['Site']
            metrics_df.at[idx, 'Model'] = mapping_dict[key]['Model']
    
    # Report results
    updated_count = metrics_df[metrics_df['Cohort'].notnull()].shape[0]
    print(f"Updated {updated_count} out of {metrics_df.shape[0]} rows")
    
    # Save to file
    metrics_df.to_csv(output_filename, index=False)
    print(f"\nMerged data saved to {output_filename}")
    
    return metrics_df

# load data 
# Load Sophie's scan database
scandataloc = 'Sophie_Data/Sophies_scan_database_20220822.csv'
scandata = pd.read_csv(scandataloc)
#print(scandata.columns)

# import patient scanner data
patient_scanner_data = pd.read_csv('DTI_Processing_Scripts/patient_scanner_data_with_timepoints.csv')
# tidy it
patient_scanner_data = patient_scanner_data.dropna(subset=['timepoint'])
print(f"Total entries in patient_scanner_data: {len(patient_scanner_data)}")
# get unique patient id and timepoint combinations
patient_scanner_data = patient_scanner_data.drop_duplicates(subset=['patient_id', 'timepoint'])
print(f"Unique patient_id and timepoint combinations: {len(patient_scanner_data)}")




# # load the metrics data from DTI_Processing_Scripts/results/all_metrics_*.csv
all_metrics_5x4vox = pd.read_csv('DTI_Processing_Scripts/results/all_metrics_5x4vox.csv')

all_metrics_5x4vox_merged = merge_scanner_info_with_metrics(
    all_metrics_5x4vox, 
    patient_scanner_data, 
    'DTI_Processing_Scripts/merged_data_5x4vox.csv'
)

