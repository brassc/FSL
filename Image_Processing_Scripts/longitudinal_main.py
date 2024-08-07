import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


# Main

# Load the ellipse data
data=pd.read_csv('Image_Processing_Scripts/ellipse_data.csv')

print(data.columns)

# get possible patient IDs
patient_ids = data['patient_id'].unique()

# array of timepoints
timepoints = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']

# patient id has all timepoints. if h_param_def data exists, put into a dictionary
patient_dict = {}
for patient_id in patient_ids:
    patient_dict[patient_id] = {}
    for timepoint in timepoints:
        subset = patient_dict[patient_id][timepoint] = data[(data['patient_id'] == patient_id) & (data['timepoint'] == timepoint)]
        if len(patient_dict[patient_id][timepoint]) > 0:
            patient_dict[patient_id][timepoint] = subset['h_param_def'].iloc[0]
        else:
            patient_dict[patient_id][timepoint] = None



# create data frame where each row is a patient id, and each column is a timepoint
df = pd.DataFrame(patient_dict)
df=df.T
print(df.head())

# Convert from wide to long format
df_reset = df.reset_index()
df_reset.columns = ['patient_id', 'ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
print(df_reset.head())

long_df=pd.melt(df_reset, id_vars=['patient_id'], var_name='timepoint', value_name='h_param_def')
# Sort by patient_id: 
#      First, convert 'timepoint' column to a categorical type with the defined order
long_df['timepoint'] = pd.Categorical(long_df['timepoint'], categories=timepoints, ordered=True)
long_df = long_df.sort_values(by=['patient_id', 'timepoint']).reset_index(drop=True)
print(long_df)

# PLOTTING
# color map
# Create a unique color map for patient_id
unique_patient_ids = long_df['patient_id'].unique()
colors = plt.get_cmap('tab10').colors  # Use 'tab10' colormap for a set of distinct colors
color_map = {pid: colors[i % len(colors)] for i, pid in enumerate(unique_patient_ids)}


# Function to determine color based on the beginning of 'unique_label'
def get_color(unique_label):
    pid = unique_label.split(' ')[0]
    return color_map.get(int(pid), 'gray')  # Default to 'gray' if pid is not found

# PLOT GROUPED BY PATIENT
plt.figure(figsize=(12, 8))
#plt.bar(long_df['timepoint'], long_df['h_param_def'] )
long_df['unique_label']=long_df['patient_id'].astype(str) + ' ' + long_df['timepoint'].astype(str)
bars=plt.bar(long_df.index, long_df['h_param_def'], color=[get_color(label) for label in long_df['unique_label']])
plt.xticks(long_df.index, long_df['unique_label'], rotation=90, fontsize=8)
plt.xlabel('Timepoint')
plt.ylabel('h_param_def')
plt.title('h_param_def by timepoint')
# Create legend
handles = [plt.Line2D([0], [0], color=color_map[pid], lw=4) for pid in unique_patient_ids]
labels = [str(pid) for pid in unique_patient_ids]
plt.legend(handles, labels, title="Patient ID")
plt.tight_layout()  # Adjust layout to fit labels
# Show the plot
plt.show()

## GROUPED BY TIMEPOINT
# Create a pivot table for plotting
pivot_df = long_df.pivot(index='timepoint', columns='patient_id', values='h_param_def')
#print(pivot_df)
# Plot
fig, ax = plt.subplots(figsize=(12, 8))
pivot_df.plot(kind='bar', ax=ax, color=[color_map[pid] for pid in pivot_df.columns]) # use same color map as before
#ax = pivot_df.plot(kind='bar', figsize=(12, 8), colormap='tab10')
plt.title('h_param_def grouped by timepoint')
plt.xlabel('Timepoint')
plt.ylabel('h_param_def')
plt.legend(title='Patient ID')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
