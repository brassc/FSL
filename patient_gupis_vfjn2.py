import pandas as pd
import numpy as np

## Load the data
DC_data = pd.read_csv('Decompression_MRI_27092024.csv') # turns out this is CT data
print(DC_data.columns) 

## go through line by line
## extract data to new df if longitudinal i.e. if First.scan.to.use. contains data
multiple_timepoint_patients_df = DC_data[DC_data['First.scan.to.use.'].notnull()].reset_index(drop=True)
## Keep only columns of interest (GUPI, First.scan.to.use., Second.scan.to.use., Third.scan.to.use., Fourth.scan.to.use., Fifth.scan.to.use.)
multiple_timepoint_patients_df = multiple_timepoint_patients_df[['GUPI', 'First.scan.to.use.', 'Second.scan.to.use', 'Third.scan.to.use', 'Fourth.scan.to.use', 'Fifth.scan.to.use', 'SurgeriesCranial.SurgeryDescCranial']]
# print(multiple_timepoint_patients_df)
## if SurgeriesCranial.SurgeryDescCranial contains 'hemi' then keep the row
multiple_timepoint_patients_df = multiple_timepoint_patients_df[multiple_timepoint_patients_df['SurgeriesCranial.SurgeryDescCranial'].str.contains('hemi')].reset_index(drop=True)
print(multiple_timepoint_patients_df)

# print GUPIs only
print("\nhemicraniectomy GUPI list:")
print(multiple_timepoint_patients_df['GUPI'].to_string(index=False))

multiple_timepoint_patients_df = multiple_timepoint_patients_df[['GUPI', 'First.scan.to.use.', 'Second.scan.to.use', 'Third.scan.to.use', 'Fourth.scan.to.use', 'Fifth.scan.to.use']]
print(multiple_timepoint_patients_df)
# save to csv
#multiple_timepoint_patients_df.to_csv('/home/cmb247/Desktop/Project_3/cmb247_longitudinal_hemicraniectomy_patients.csv', index=False)



### NOW CROSS REFERENCE WITH CENTER_TBI OLIVIA DATA (all patients with MRI scans)
# Load the data
mri_data = pd.read_csv('CENTER_TBI_Scans(Sheet1).csv')
print(mri_data.columns)
# Get the GUPIs of the patients with MRI scans
mri_data_gupis = mri_data['GUPI'].unique()
# Get the GUPIs of the patients with decompression surgery
#decompression_data_gupis = multiple_timepoint_patients_df['GUPI'].unique()
decompression_data_gupis = DC_data['GUPI'].unique()
# Get the GUPIs of the patients with decompression surgery and MRI scans
try:   
    common_gupis = np.intersect1d(mri_data_gupis, decompression_data_gupis)
except ValueError:
    common_gupis = []
    print("No patients with MRI scans and decompression surgery")
    import sys
    sys.exit(1)

print("\nPatients with MRI scan(s) and decompression surgery:")
print(common_gupis)

# go through mri_data and extract the rows with GUPIs in common_gupis but only the Scan_ID column
common_gupis_df = mri_data[mri_data['GUPI'].isin(common_gupis)].reset_index(drop=True)
common_gupis_df = common_gupis_df[['GUPI', 'Scan_ID']]
print(common_gupis_df)

# remove the rows where the GUPI appears only once
# get duplicate GUPIs
duplicate_gupis = common_gupis_df['GUPI'].value_counts()
duplicate_gupis = duplicate_gupis[duplicate_gupis > 1].index
# Filter for those GUPIS
common_gupis_df = common_gupis_df[common_gupis_df['GUPI'].isin(duplicate_gupis)].reset_index(drop=True)
print(common_gupis_df)






