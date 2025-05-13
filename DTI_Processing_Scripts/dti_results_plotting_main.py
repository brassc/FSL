import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import seaborn as sns
import sys
import os
import statsmodels.stats.multitest as smm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Image_Processing_Scripts.longitudinal_main import map_timepoint_to_string
from Image_Processing_Scripts.set_publication_style import set_publication_style

import rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import PackageNotInstalledError

# Activate pandas to R conversion
pandas2ri.activate()

# First, install required R packages if not already installed
utils = importr('utils')
utils.chooseCRANmirror(ind=1)  # Choose the first CRAN mirror

# Check and install required packages
packnames = ('lme4', 'emmeans', 'pbkrtest', 'lmerTest')
names_to_install = [x for x in packnames if not rpy2.robjects.packages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

# Import necessary R packages
base = importr('base')
lme4 = importr('lme4')
emmeans = importr('emmeans')


# Set publication style for matplotlib
set_publication_style()

if __name__ == '__main__':
    print('running dti_results_plotting_main.py')
    
    # Load the data 
    data_5x4vox=pd.read_csv('DTI_Processing_Scripts/merged_data_5x4vox_NEW.csv')
    data_5x4vox['patient_id'] = data_5x4vox['patient_id'].astype(str)
    timepoints = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
    string_mask = data_5x4vox['timepoint'].isin(timepoints)
    numeric_mask = ~string_mask & data_5x4vox['timepoint'].apply(lambda x: pd.notnull(pd.to_numeric(x, errors='coerce')))

    # Convert numeric values to their appropriate string representations
    for idx in data_5x4vox[numeric_mask].index:
        try:
            numeric_value = float(data_5x4vox.loc[idx, 'timepoint'])
            data_5x4vox.loc[idx, 'timepoint'] = map_timepoint_to_string(numeric_value)
        except (ValueError, TypeError):
            continue
    
    #data_5x4vox['timepoint'] = data_5x4vox['timepoint'].apply(map_timepoint_to_string) # convert to string category
    #print(f"data_5x4vox:\n{data_5x4vox}")


     
    # Save original timepoint values before recategorization (if needed for reference)
    data_5x4vox['original_timepoint'] = data_5x4vox['timepoint']
    
    # Now apply the timepoint recategorization based on Days_since_injury
    # Define the ranges and labels for recategorization
    ranges = [(0, 2), (2, 8), (8, 42), (42, 179), (179, 278), (278, 540), (540, 500000)]
    labels = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
    
    # Only recategorize rows where Days_since_injury is not null
    mask = data_5x4vox['Days_since_injury'].notnull()
    if mask.any():
        data_5x4vox.loc[mask, 'timepoint'] = pd.cut(
            data_5x4vox.loc[mask, 'Days_since_injury'], 
            bins=[0, 2, 8, 42, 179, 278, 540, 500000], 
            labels=labels
        )

    # sort data according to patient id then timepoint

    def get_sort_key(patient_id):
        try:
            return (0, int(patient_id))  # Numeric IDs first, sorted numerically
        except ValueError:
            return (1, patient_id)       # Alphanumeric IDs second, sorted alphabetically

    # Create a sort key column
    data_5x4vox['sort_key'] = data_5x4vox['patient_id'].apply(get_sort_key)
    data_5x4vox['timepoint_order'] = data_5x4vox['timepoint'].apply(lambda x: timepoints.index(x) if x in timepoints else 999)
    
    # Sort the dataframe by patient_id, then by the position of timepoint in our list
    data_5x4vox = data_5x4vox.sort_values(by=['sort_key', 'timepoint_order'])
    data_5x4vox = data_5x4vox.drop(['sort_key', 'timepoint_order'], axis=1)  # remove sorting column
    
    # Now data_5x4vox has been recategorized based on Days_since_injury, exactly the same as the deformation analysis
    


