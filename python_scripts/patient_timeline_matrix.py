import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns


data=pd.read_csv('/Users/charlottebrass/repos/FSL/patient_timeline_map.csv')

# Fill empty cells with NaN
data = data.fillna(0)

# Ignoring the first and last columns from the data
trimmed_data = data.drop(data.columns[[0, -1]], axis=1)


for col in trimmed_data.columns[:8]:  # Selecting the first 8 columns
    trimmed_data[col] = trimmed_data[col].astype(int)

# Create a copy of the trimmed data
modified_data = trimmed_data.copy()

# Set a distinct value for the last 7 rows (assuming 2 is not present in original data)
modified_data.iloc[-7:, :] = modified_data.iloc[-7:, :] * 2  # For example, setting to 2
print(modified_data)

# Define a custom color map (binary + blue for value 2)
cmap = mcolors.ListedColormap(['white', 'black', 'blue'])
bounds = [0, 1, 2, 3]
norm = mcolors.BoundaryNorm(bounds, cmap.N)


# Create a figure and axis
plt.figure(figsize=(trimmed_data.shape[1]*0.5, trimmed_data.shape[1]))

# Use imshow to create the matrix plot
plt.imshow(modified_data, cmap=cmap, norm=norm, interpolation='none')

plt.tick_params(
    axis='both',          # changes apply to both x and y axis
    which='both',         # both major and minor ticks are affected
    bottom=False,         # ticks along the bottom edge are off
    top=False,            # ticks along the top edge are off
    left=False,           # ticks along the left edge are off
    right=False,          # ticks along the right edge are off
    labelbottom=True,    # labels along the bottom edge are on
    labelleft=True       # labels along the left edge are on
)
y_labels = data.iloc[:, 0].astype(int).astype(str)
# Replace '0' with a blank space
y_labels = y_labels.replace('0', ' ')

plt.yticks(ticks=range(len(y_labels)), labels=y_labels)
x_labels = data.columns[1:].drop(data.columns[-1])
plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=90) 
plt.xlabel("Timescale", labelpad=15)  # Increase labelpad for xlabel
plt.ylabel("Patient ID", labelpad=15)  # Increase labelpad for ylabel
plt.title("Patient ID Timescale Matrix", pad=20)  # Increase pad for title

# Create patches for the legend
blue_patch = mpatches.Patch(color='blue', label='bifrontal')
black_patch = mpatches.Patch(color='black', label='hemi')

# Add the legend to the plot
#plt.legend(handles=[blue_patch, black_patch])
plt.legend(handles=[blue_patch, black_patch], loc='upper center', bbox_to_anchor=(0.5, -0.275), fancybox=False, shadow=False, ncol=2)

# Adjust the subplot parameters to add more white space at the bottom
plt.subplots_adjust(bottom=0.25)  # Increase the bottom margin

# Update save location as needed
plt.savefig('python_scripts/patient_timeline_matrix.png')

plt.show()


