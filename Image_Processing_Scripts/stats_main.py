import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
import sys

# Load the data (note this data does not contain all timepoints w NaN value if not exist - only contains timepoints w data per original data)
data = pd.read_csv('Image_Processing_Scripts/area_data.csv')
# drop cols from dataframe: remove area_def, area_ref and side
new_df=data.drop(columns='area_def', axis=1)
new_df=new_df.drop(columns='area_ref', axis=1)
new_df=new_df.drop(columns='side', axis=1)

# Drop rows where area_diff is NaN
new_df = new_df.dropna(subset=['area_diff']) # there should be no NaN values in area_diff, but just in case

# Ensure categorical values are categorical
new_df['timepoint']=pd.Categorical(new_df['timepoint'], categories=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo'])

# Fit the model
model = smf.mixedlm("area_diff ~ timepoint", new_df, groups=new_df["patient_id"], re_formula="~1")
result = model.fit()
# Print the summary
print(result.summary())

# Scan by 'first' scan, use this as category for comparison
new_df=new_df.sort_values(by=['patient_id','timepoint'])
new_df['first_scan']=None
print(new_df.head())
# Add a column to the dataframe that indicates the first scan for each patient. first scan is either fast or ultrafast
new_df['first_scan']=new_df.groupby('patient_id')['timepoint'].transform('first')
new_df['scan_type']=new_df.apply(lambda row: 'first' if row['timepoint']==row['first_scan'] 
                             else 'fast' if row['timepoint'] == 'fast'
                             else 'acute' if row['timepoint'] == 'acute' 
                             else '3mo' if row['timepoint'] == '3mo' 
                             else '6mo' if row['timepoint'] == '6mo' 
                             else '12mo' if row['timepoint'] == '12mo' 
                             else '24mo' if row['timepoint'] == '24mo'
                             else 'other', axis=1)
print(new_df.head())    


model = smf.mixedlm("area_diff ~ C(scan_type)", new_df, groups=new_df["patient_id"], re_formula="~1")
result = model.fit()

# Print the summary
print(result.summary())
