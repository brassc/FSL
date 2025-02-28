import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy import stats


# Filenames
input_ellipse_filename = 'ellipse_area_data.csv' # input
input_batch2_elllipse_filename = 'batch2_ellipse_area_data.csv' # input
input_contour_filename = 'area_data.csv' # input
input_batch2_contour_filename = 'batch2_area_data.csv' # input

# Import data from CSV files
print("Loading data from CSV files...")
ellipse_data = pd.read_csv(f'Image_Processing_Scripts/{input_ellipse_filename}')
batch2_ellipse_data = pd.read_csv(f'Image_Processing_Scripts/{input_batch2_elllipse_filename}')
contour_data = pd.read_csv(f'Image_Processing_Scripts/{input_contour_filename}')
batch2_contour_data = pd.read_csv(f'Image_Processing_Scripts/{input_batch2_contour_filename}')

#combine ellipse_data and batch2_ellipse_data
combined_ellipse_data = pd.concat([ellipse_data, batch2_ellipse_data], ignore_index=True)
# combine contour_data and batch2_contour_data
combined_contour_data = pd.concat([contour_data, batch2_contour_data], ignore_index=True)

# plot ellipse data vs contour data - area_diff vs area_diff



# Create a filter to remove outliers
# Assuming the outlier is around (4000, 4000)
x_threshold = 2500  # Adjust as needed
y_threshold = 2500  # Adjust as needed

# Create filtered data
mask = (combined_contour_data['area_diff'] < x_threshold) & (combined_ellipse_data['ellipse_area_diff'] < y_threshold)
filtered_x = combined_contour_data['area_diff'][mask]
filtered_y = combined_ellipse_data['ellipse_area_diff'][mask]

# set plot area to be square 8 x 8


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

plt.figure(figsize=(8, 6), dpi=150)
#plt.scatter(combined_contour_data['area_diff'], combined_ellipse_data['ellipse_area_diff'], color='blue', s=10)
plt.scatter(filtered_x, filtered_y, color='blue', s=10, alpha=0.7, edgecolor='navy', linewidth=0.5)

# Add reference lines for x=0 and y=0
#plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
#plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

# Set axis limits based on actual data range plus small margin
x_margin = (filtered_x.max() - filtered_x.min()) * 0.05
y_margin = (filtered_y.max() - filtered_y.min()) * 0.05

plt.xlim(filtered_x.min() - x_margin, filtered_x.max() + x_margin)
plt.ylim(filtered_y.min() - y_margin, filtered_y.max() + y_margin)


# Add grid
plt.grid(True, alpha=0.3)
# # Ensure y-axis goes to at least 4000
# ymin, ymax = plt.ylim()
# plt.ylim(ymin, max(ymax, 2000))
# # Ensure x-axis goes to at least 4000
# xmin, xmax = plt.ylim()
# plt.xlim(ymin, max(xmax, 2000))

# Calculate and add line of best fit in red
x = filtered_x #combined_contour_data['area_diff']
y = filtered_y #combined_ellipse_data['ellipse_area_diff']
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
line_x = np.array([min(x), max(x)])
line_y = slope * line_x + intercept
plt.plot(line_x, line_y, color='red', linewidth=1)
# get gradient of line of best fit
# Create a statistics box with the regression details

# Format p-value properly
if p_value < 0.0001:
    p_text = "p < 0.0001"
else:
    p_text = f"p = {p_value:.4f}"

stats_box = (
    f"$\\mathrm{{Slope}} = {slope:.2f} \\pm {std_err:.2f}$\n"
    f"$R^2 = {r_value**2:.2f}$\n"
    f"$R = {r_value:.2f}$\n"
    f"{p_text}"
)

# Add statistics text in a box
plt.text(0.05, 0.95, stats_box, transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
         fontsize=12, verticalalignment='top')


plt.gca().set_aspect('equal', adjustable='box')
# Add labels with LaTeX formatting
plt.xlabel(r'Contour Area Difference, $\Delta A_c$ [mm$^2$]', fontsize=14)
plt.ylabel(r'Ellipse Area Difference, $\Delta A_e$ [mm$^2$]', fontsize=14)
plt.title('Ellipse Area Difference vs Contour Area Difference', fontsize=16, fontweight='bold')
plt.savefig('Image_Processing_Scripts/plots/ellipse_vs_contour_area_diff.png', dpi=150)
plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/ellipse_vs_contour_area_diff.pdf', dpi=300)
plt.close()

