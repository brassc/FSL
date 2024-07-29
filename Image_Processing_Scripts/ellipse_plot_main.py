import pandas as pd
import numpy as np


## MAIN SCRIPT TO PLOT ELLIPSE FORM
data = pd.read_csv('Image_Processing_Scripts/data_entries.csv')
print(data.columns)
print(data.iloc[0, 2])

h_def=data['deformed_contour_x']
v_def=data['deformed_contour_y']
h_ref=data['reflected_contour_x']
v_ref=data['reflected_contour_y']
print("h_def: ", h_def.iloc[0])

hv_df = pd.DataFrame({'h_def':h_def, 'v_def':v_def, 'h_ref':h_ref, 'v_ref':v_ref})
#print(hv_df.iloc[0,1])
total_df=pd.concat([data, hv_df], axis=1)

# Transform data points such that posterior point lies on (0, 0) and anterior lies on y=0 (x axis) (for R side craniectomy) 
#   or anterior point lies on (0,0) and posterior lies on y=0 (x axis) (for L side craniectomy)



# Find average of start and end y, position about center

# Make y values positive if necessary

# fit ellipse using least squares method

# Find change in area between two ellipses

# Plot change in area over time for each patient (x axis: time, y axis: area)

# Reverse transform data points, save to df / .csv

# Plot on image.