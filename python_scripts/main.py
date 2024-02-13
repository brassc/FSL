#standard libraries
import pandas as pd

#custom functions
from data_filter import data_filter
from read_data import read_data
from plots import plot_matrix

# READ DATA FROM CSV
df = read_data() # default parameter is csv_loc='/Users/charlottebrass/repos/FSL/patient_timeline_map.csv'

# FILTER DATA TO EXCLUDE PATIENTS WITH ONLY 1 SCAN TIMEPOINT
filtered_df = data_filter(df)

# PLOT MATRIX USING THIS FILTERED DATA. SAVE TO FILE AS 'filtered_patient_timeline_matrix.png'
plot_matrix(filtered_df, filename='filtered_patient_timeline_matrix.png')

