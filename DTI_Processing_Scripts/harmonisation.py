import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuroCombat import neuroCombat

# load data 
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
batch = pd.concat([batch1, batch2], ignore_index=True)
# remove excluded patients if excluded == 1
batch = batch[batch['excluded'] != 1].reset_index(drop=True)
print(batch.columns)
n_scans=len(batch)
n_patients=len(batch['patient_id'].unique())


# print batch
pd.set_option('display.max_rows', None)
print(batch)
pd.set_option('display.max_rows', pd.options.display.max_rows) # reset to default



