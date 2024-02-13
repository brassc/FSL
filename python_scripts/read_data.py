
import pandas as pd



def read_data(csv_loc='/Users/charlottebrass/repos/FSL/patient_timeline_map.csv'):
    # THIS FUNCTION READS DATA FROM SPECIFIED OR DEFAULT LOCATION, FILLS EMPTY CELLS WITH NaN 
    # AND DROPS THE LAST COLUMN FROM THE DATA [IN THIS CASE, THESE ARE CHARACTER LETTERS]. IT ALSO CONVERTS
    # DATA TO INT VALUES. NOTE PATIENT ID'S ARE RETAINED. 
    # default location is set to '/Users/charlottebrass/repos/FSL/patient_timeline_map.csv'

    # Read data in 
    data=pd.read_csv(csv_loc)

    # Fill empty cells with NaN
    data = data.fillna(0)

    # Drop the last column from the data
    trimmed_data = data.iloc[:, :-1]

    # Convert to int
    for col in trimmed_data.columns[:8]:  # Selecting the first 8 columns
        trimmed_data[col] = trimmed_data[col].astype(int)

    # Create a copy of the trimmed data
    df = trimmed_data.copy()

    return df



