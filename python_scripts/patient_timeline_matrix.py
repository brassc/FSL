import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data=pd.read_csv('/Users/charlottebrass/repos/FSL/patient_timeline_map.csv')

# Fill empty cells with NaN
data = data.fillna(0)

# Ignoring the first and last columns from the data
trimmed_data = data.drop(data.columns[[0, -1]], axis=1)


for col in trimmed_data.columns[:8]:  # Selecting the first 8 columns
    trimmed_data[col] = trimmed_data[col].astype(int)

# Print the matrix
#print(trimmed_data[2:5])
# Converting the data to a binary matrix

# Create a figure and axis
plt.figure(figsize=(trimmed_data.shape[0], trimmed_data.shape[1]))

# Use imshow to create the matrix plot
plt.imshow(trimmed_data, cmap='binary', interpolation='none')
y_labels = data.iloc[:, 0].astype(int)
plt.yticks(ticks=range(len(y_labels)), labels=y_labels)
x_labels = data.columns[1:].drop(data.columns[-1])
plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=90) 
plt.xlabel("Timescale")
plt.ylabel("Patient ID")
plt.title("Patient ID Timescale Matrix")
plt.show()

