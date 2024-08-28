import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import LinearSegmentedColormap, rgb2hex

def get_dictionary(data, patient_ids, timepoints, subset_name='h_param_def'):
    patient_dict = {}
    for patient_id in patient_ids:
        patient_dict[patient_id] = {}
        for timepoint in timepoints:
            subset = patient_dict[patient_id][timepoint] = data[(data['patient_id'] == patient_id) & (data['timepoint'] == timepoint)] # subset = patient_dict[patient_id][timepoint] = 
            if len(patient_dict[patient_id][timepoint]) > 0:
                patient_dict[patient_id][timepoint] = subset[subset_name].iloc[0]
            else:
                patient_dict[patient_id][timepoint] = None
    return patient_dict

def convert_dict_to_long_df(data_dict, timepoints, value_name='h_param_def'):
    # Convert from wide to long format
    df = pd.DataFrame(data_dict)
    df = df.T
    df_reset = df.reset_index()
    df_reset.columns = ['patient_id'] + timepoints
    long_df = pd.melt(df_reset, id_vars=['patient_id'], var_name='timepoint', value_name=value_name)
    # Sort by patient_id
    long_df['timepoint'] = pd.Categorical(long_df['timepoint'], categories=timepoints, ordered=True)
    long_df = long_df.sort_values(by=['patient_id', 'timepoint']).reset_index(drop=True)
    return long_df

def plot_longitudinal_data(long_df, name):
    # color map
    # Create a unique color map for patient_id
    unique_patient_ids = long_df['patient_id'].unique()
    n=len(unique_patient_ids) # number of unique patient id's for color map
    #colors = plt.get_cmap('tab10').colors  # Use 'tab10' colormap for a set of distinct colors
    # colours to use:
    base_colors = ['red', 'cyan', 'yellow', 'magenta', 'brown', 'lightblue', 'orange']

    def create_hex_color_map_from_cmap(cmap_name,n):
        cmap=plt.get_cmap(cmap_name)
        # Convert the colormap to a list of hex colors
        colors=cmap(np.linspace(0,1,n))
        hex_colors = [rgb2hex(color) for color in colors]
        return hex_colors
    
    def create_hex_color_map_custom(base_colors,n):
        # Create a custom colormap with the base colors
        cmap = LinearSegmentedColormap.from_list('tab20', base_colors, N=n)
        # Convert the colormap to a list of hex colors
        colors=cmap(np.linspace(0,1,n))
        hex_colors = [rgb2hex(color) for color in colors]
        return hex_colors

    
    hex_color_map=create_hex_color_map_from_cmap('tab20',n)
    hex_color_map=create_hex_color_map_custom(base_colors,n)
    print(hex_color_map)
    

    #cmap=plt.get_cmap('tab10')
    #colors = cmap(np.linspace(0,1,n)) # n is the number of colours
    color_map = {pid: hex_color_map[i] for i, pid in enumerate(unique_patient_ids)}

    # Function to determine color based on the beginning of 'unique_label'
    def get_color(unique_label):
        pid = unique_label.split(' ')[0]
        return color_map.get(int(pid), 'gray')  # Default to 'gray' if pid is not found

    # PLOT GROUPED BY PATIENT
    plt.figure(figsize=(12, 8))
    #plt.bar(long_df['timepoint'], long_df['h_param_def'] )
    long_df['unique_label']=long_df['patient_id'].astype(str) + ' ' + long_df['timepoint'].astype(str)
    bars=plt.bar(long_df.index, long_df[name], color=[get_color(label) for label in long_df['unique_label']])
    plt.xticks(long_df.index, long_df['unique_label'], rotation=90, fontsize=8)
    plt.xlabel('Timepoint')
    plt.ylabel(name)
    plt.title(f"{name} by patient")
    # Create legend
    handles = [plt.Line2D([0], [0], color=color_map[pid], lw=4) for pid in unique_patient_ids]
    labels = [str(pid) for pid in unique_patient_ids]
    plt.legend(handles, labels, title="Patient ID")
    plt.tight_layout()  # Adjust layout to fit labels
    # Show the plot
    if name == 'h_param_def' or name == 'h_param_ref':
        plt.ylim([0, 1.0])
    #plt.ylim([0, 1.0])
    plt.show()

  
    ## GROUPED BY TIMEPOINT
    # Create a pivot table for plotting
    pivot_df = long_df.pivot(index='timepoint', columns='patient_id', values=name)
    #print(pivot_df)
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_df.plot(kind='bar', ax=ax, color=[color_map[pid] for pid in pivot_df.columns]) # use same color map as before
    #ax = pivot_df.plot(kind='bar', figsize=(12, 8), colormap='tab10')
    plt.title(f"{name} grouped by timepoint")
    plt.xlabel('Timepoint')
    plt.ylabel(name)
    plt.legend(title='Patient ID')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if name == 'h_param_def' or name == 'h_param_ref':
        plt.ylim([0, 1.0])
    plt.show()

    return 0



# Main

# Load the ellipse data
data=pd.read_csv('Image_Processing_Scripts/ellipse_data.csv')
print(data.columns)
area_data=pd.read_csv('Image_Processing_Scripts/area_data.csv')
print(area_data.columns)

# get possible patient IDs
patient_ids = data['patient_id'].unique()
n = len(patient_ids) # number of colours for plotting

# array of timepoints
timepoints = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']

# h param plots
h_param_def_dict = get_dictionary(data, patient_ids, timepoints, subset_name='h_param_def')
h_param_ref_dict = get_dictionary(data, patient_ids, timepoints, subset_name='h_param_ref')

def_df = convert_dict_to_long_df(h_param_def_dict, timepoints, value_name='h_param_def')
ref_df = convert_dict_to_long_df(h_param_ref_dict, timepoints, value_name='h_param_ref')

#plot_longitudinal_data(def_df, name='h_param_def')
#plot_longitudinal_data(ref_df, name='h_param_ref')

# a param plots (note a_param_ref should be the same as a_param_def)
a_param_def_dict = get_dictionary(data, patient_ids, timepoints, subset_name='a_param_def')
#a_param_ref_dict = get_dictionary(data, patient_ids, timepoints, subset_name='a_param_ref')

defa_df = convert_dict_to_long_df(a_param_def_dict, timepoints, value_name='a_param_def')
#refa_df = convert_dict_to_long_df(a_param_ref_dict, timepoints, value_name='a_param_ref')

#plot_longitudinal_data(defa_df, name='a_param_def')
#plot_longitudinal_data(refa_df, name='a_param_ref')

# Area plots [area calculated prior to rotation from orienting ellipse_data.csv '<>_<>ef_tr' columns in area_main.py]
area_diff_dict = get_dictionary(area_data, patient_ids, timepoints, subset_name='area_diff')
area_diff_df = convert_dict_to_long_df(area_diff_dict, timepoints, value_name='area_diff')
#plot_longitudinal_data(area_diff_df, name='area_diff')

# Line plots superimposed

print(area_diff_df)
# convert timepoints to a numerical array 0 to len(timepoints)
timepoints_num = np.arange(len(timepoints))

# recall patient_ids = data['patient_id'].unique() have already been collected
for patient_id in patient_ids:
    patient_subset = area_diff_df[area_diff_df['patient_id'] == patient_id]
    print(patient_subset['area_diff'])
    # convert patient_subset['area_diff'] to a numpy array for plotting
    area_diff_subset = np.array(patient_subset['area_diff'])
    valid_indices = ~np.isnan(area_diff_subset)
    area_diff_subset_valid = area_diff_subset[valid_indices]
    timepoints_num_valid = timepoints_num[valid_indices]
    plt.plot(timepoints_num_valid, area_diff_subset_valid, label=patient_id)

plt.xlim([0, len(timepoints)-1])
plt.legend(title='Patient ID')
plt.show()