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

print(total_df.columns)
print(total_df.iloc[0,2])
print('total_df_h_def', total_df.iloc[0,6])