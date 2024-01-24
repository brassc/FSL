import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.gridspec as gridspec


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




# MATRIX ONLY 

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


# MATRIX AND BAR PLOT TOTALS

# Create a figure with a grid layout for subplots
fig = plt.figure(figsize=(4, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[8, 1])  # 2 rows, 1 column, height ratios for subplots

# Create the main axis for the matrix plot in the first row
ax1 = plt.subplot(gs[0])

# Use imshow to create the matrix plot
ax1.imshow(modified_data, cmap=cmap, norm=norm, interpolation='none')

# Customize the existing plot
ax1.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=True,
    labelleft=True
)

y_labels = data.iloc[:, 0].astype(int).astype(str)
y_labels = y_labels.replace('0', ' ')
ax1.set_yticks(ticks=range(len(y_labels)))
ax1.set_yticklabels(y_labels, fontsize=8)

x_labels = data.columns[1:].drop(data.columns[-1])
ax1.set_xticks(ticks=range(len(x_labels)))
#ax1.set_xticklabels(x_labels, rotation=90)
# Omit x-axis labels for the second bar chart
ax1.set_xticklabels([])
ax1.set_ylabel("Patient ID", labelpad=15)
ax1.set_title("Patient ID Timescale Matrix", pad=20)

blue_patch = mpatches.Patch(color='blue', label='bifrontal')
black_patch = mpatches.Patch(color='black', label='hemi')
ax1.legend(handles=[blue_patch, black_patch], loc='upper center', bbox_to_anchor=(0.5, -0.65), fancybox=False, shadow=False, ncol=2, fontsize=8)

# Calculate the column sums of trimmed_data
column_sums = trimmed_data.sum()
bifrontal_sums = trimmed_data.iloc[-7:, :].sum()
hemi_sums = trimmed_data.iloc[:25, :].sum()
barchart_df = pd.concat([bifrontal_sums, hemi_sums], axis=1, keys=['Bifrontal', 'Hemi'])

# Create a new axis for the bar chart in the second row
ax2 = plt.subplot(gs[1])

# Create the bar chart with the same width as the matrix plot
bar_width = 4/len(x_labels)  # Adjust the bar width as needed
#ax2.bar(x_labels, column_sums, color='gray', width=bar_width)
ax2.bar(x_labels, barchart_df['Bifrontal'], color='blue', width=bar_width, label='Bifrontal')
ax2.bar(x_labels, barchart_df['Hemi'], color='black', width=bar_width, bottom=barchart_df['Bifrontal'], label='Hemi')
ax2.set_ylabel('Total')
ax2.set_xticks(ticks=range(len(x_labels)))
ax2.set_xticklabels(x_labels, rotation=90, fontsize=8)
ax2.tick_params(axis='x', length=0)
ax2.set_xlabel("Timescale", labelpad=20)

# Adjust subplot parameters for the main plot to add more white space at the bottom
plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(bottom=0.2)
plt.subplots_adjust(top=0.9) 

# Draw the figure to update the renderer and get the correct bounding boxes
fig.canvas.draw()

# Get the position of the first subplot after rendering the figure
pos1 = ax1.get_position()

# Set the position of the second subplot to match the width of the first subplot
pos2 = [pos1.x0, 0.2, pos1.width, 0.15] # The y0 and height values here are placeholders and should be adjusted as needed
ax2.set_position(pos2)




# Redraw the figure to apply the new changes
fig.canvas.draw()

# Update save location as needed
plt.savefig('python_scripts/patient_timeline_matrix_and_sums.png')

# Show the combined figure
plt.show()

