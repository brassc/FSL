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
print(batch.columns)

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


