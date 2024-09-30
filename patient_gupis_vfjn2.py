import pandas as pd
import numpy as np

## Load the data
DC_data = pd.read_csv('/home/cmb247/Desktop/Project_3/Decompression_MRI_27092024.csv')
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


