import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import seaborn as sns
import sys
from scipy.stats import mannwhitneyu
import statsmodels.stats.multitest as smm
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from longitudinal_main import map_timepoint_to_string


print('running stats_main.py')
# Load the data (note this data does not contain all timepoints w NaN value if not exist - only contains timepoints w data per original data)
batch1_data = pd.read_csv('Image_Processing_Scripts/area_data.csv')
batch2_data = pd.read_csv('Image_Processing_Scripts/batch2_area_data.csv')
batch2_data['timepoint'] = batch2_data['timepoint'].apply(map_timepoint_to_string) # convert to string category
# Combine the two batches
data = pd.concat([batch1_data, batch2_data], ignore_index=True)

# sort data according to patient id then timepoint
# Sort data by patient_id numerically when possible
def get_sort_key(patient_id):
    try:
        return (0, int(patient_id))  # Numeric IDs first, sorted numerically
    except ValueError:
        return (1, patient_id)       # Alphanumeric IDs second, sorted alphabetically

# Create a sort key column
data['sort_key'] = data['patient_id'].apply(get_sort_key)


timepoints = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
data['timepoint_order'] = data['timepoint'].apply(lambda x: timepoints.index(x) if x in timepoints else 999)
# Sort the dataframe by patient_id, then by the position of timepoint in our list
data = data.sort_values(by=['sort_key', 'timepoint_order'])
data = data.drop(['sort_key', 'timepoint_order'], axis=1) # remove sorting column
print(data)


# drop cols from dataframe: remove area_def, area_ref and side
new_df=data.drop(columns='area_def', axis=1)
new_df=new_df.drop(columns='area_ref', axis=1)
new_df=new_df.drop(columns='side', axis=1)

# Drop rows where area_diff is NaN
new_df = new_df.dropna(subset=['area_diff']) # there should be no NaN values in area_diff, but just in case
print('rows dropped')
# Ensure categorical values are categorical
print('ensuring categorical values are categorical')
new_df['timepoint']=pd.Categorical(new_df['timepoint'], categories=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo'])
print(new_df.head())


###### NEW CODE 2025-03-03 #######
# Check for duplicates
print("Checking for duplicates:")
dupes = new_df.duplicated(subset=['patient_id', 'timepoint'], keep=False)
print(f"Number of duplicate entries: {dupes.sum()}")
if dupes.sum() > 0:
    print("Sample of duplicates:")
    print(new_df[dupes])# Using all available data


pivoted_data = new_df.pivot(index='patient_id', columns='timepoint', values='area_diff')
pivoted_data = pivoted_data.rename_axis(None, axis=1)

# List of all timepoints
timepoints = pivoted_data.columns.tolist()

# Initialise dictionary to store results
results_all_pairs = {}

# Perform pairwise comparisons among all timepoints

# Function to perform paired t-test between two time points
def paired_ttest(data, time1, time2):
    # Get data for both time points, excluding rows with missing values in either
    paired_data = data[[time1, time2]].dropna()
    
    if len(paired_data) < 3:
        return {
            'comparison': f"{time1} vs {time2}",
            'time1': time1,
            'time2': time2,
            'n_pairs': len(paired_data),
            'mean_diff': np.nan,
            'std_diff': np.nan,
            't_statistic': np.nan,
            'p_value': np.nan,
            'sufficient_data': False
        }
    
    # Perform paired t-test
    t_stat, p_val = stats.ttest_rel(paired_data[time1], paired_data[time2])
    
    # Calculate mean and std of differences
    diff = paired_data[time1] - paired_data[time2]
    
    return {
        'comparison': f"{time1} vs {time2}",
        'time1': time1,
        'time2': time2,
        'n_pairs': len(paired_data),
        'mean_diff': diff.mean(),
        'std_diff': diff.std(),
        't_statistic': t_stat,
        'p_value': p_val,
        'sufficient_data': True
    }

# Generate all possible pairs of timepoints
pairs = list(combinations(timepoints, 2))

# Run paired t-tests
ttest_results = []
for time1, time2 in pairs:
    ttest_result = paired_ttest(pivoted_data, time1, time2)
    ttest_results.append(ttest_result)
# Convert results to DataFrame
results_df = pd.DataFrame(ttest_results)
#print(results_df)

# Filter to show only pairs with sufficient data
valid_results = results_df[results_df['sufficient_data'] == True].copy()

if len(valid_results) > 0:
    # Apply Holm-Bonferroni correction to p-values
    if len(valid_results) > 1:  # Only apply if there are multiple tests
        p_values = valid_results['p_value'].values
        rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')
        valid_results['p_corrected'] = p_corrected
        valid_results['significant'] = rejected
    else:
        # If only one test, no correction needed
        valid_results['p_corrected'] = valid_results['p_value']
        valid_results['significant'] = valid_results['p_value'] < 0.05
    
    # Sort by corrected p-value
    valid_results = valid_results.sort_values('p_corrected')
    
    # Print results
    print("\nPaired t-test Results with Holm-Bonferroni correction:")
    print(valid_results[['comparison', 'n_pairs', 'mean_diff', 't_statistic', 'p_value', 'p_corrected', 'significant']])

    
    # Store results in a dictionary for later use
    results_all_pairs = {}
    for _, row in valid_results.iterrows():
        key = (row['time1'], row['time2'])
        results_all_pairs[key] = {
            'n_pairs': row['n_pairs'],
            'mean_diff': row['mean_diff'],
            't_statistic': row['t_statistic'],
            'p_value': row['p_value'],
            'p_corrected': row['p_corrected'],
            'significant': row['significant']
        }
else:
    print("No pairs have sufficient data for paired t-test.")
    results_all_pairs = {}


# Add a column to pivoted_data to distinguish between patients with and without deformation (yes = 1)
pivoted_data['deformed'] = pivoted_data.isna().sum(axis=1) < len(timepoints)
print(pivoted_data['deformed'])
sys.exit()




### Visualisations

"""
# 1. Data availability matrix
availability_matrix = pd.DataFrame(index=timepoints, columns=timepoints, dtype=float)

# fill the matrix with counts
for time1 in timepoints:
    for time2 in timepoints:
        if time1 == time2:
            # Diagonal: number of non-missing values for this time point
            availability_matrix.loc[time1, time2] = pivoted_data[time1].notna().sum()
        else:
            # Off-diagonal: number of patients with data for both time points
            common_data = pivoted_data[[time1, time2]].dropna()
            availability_matrix.loc[time1, time2] = len(common_data)

# Verify the data types before plotting
print("Data types in availability_matrix:")
print(availability_matrix.dtypes)
print("\nSample of availability_matrix:")
print(availability_matrix.head())

# Explicitly convert matrix to float if needed
availability_matrix = availability_matrix.astype(float)

# Visualize the data availability
plt.figure(figsize=(10, 8))
sns.heatmap(availability_matrix, annot=True, cmap="YlGnBu", fmt='g')
plt.title('Data Availability Matrix (number of patients)')
plt.grid(False)
plt.tight_layout()
plt.savefig('Image_Processing_Scripts/data_availability.png')
plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/data_availability.png', dpi=600)
plt.close()

# 2. Create Significance Matrix
# 1. Significance Matrix
# ---------------------
# Create a matrix of corrected p-values
significance_matrix = pd.DataFrame(index=timepoints, columns=timepoints, dtype=float)
significance_matrix.fillna(1.0, inplace=True)  # Default p-value = 1 (not significant)

# Fill in the values from our results
for _, row in valid_results.iterrows():
    significance_matrix.loc[row['time1'], row['time2']] = row['p_corrected']
    significance_matrix.loc[row['time2'], row['time1']] = row['p_corrected']  # Mirror since it's symmetric

# Set diagonal to NaN for better visualization
for time in timepoints:
    significance_matrix.loc[time, time] = np.nan

# Visualize the significance matrix
plt.figure(figsize=(10, 8))
mask = np.isnan(significance_matrix)
cmap = sns.diverging_palette(240, 10, as_cmap=True)

# Create heatmap without grid lines
heatmap = sns.heatmap(
    significance_matrix, 
    annot=True,           # Show numbers in cells
    cmap=cmap,            # Color map
    mask=mask,            # Mask diagonal values
    vmin=0, vmax=1,     # Set color scale range
    center=0.25,          # Center color scale
    fmt='.3f',            # Format as floating point with 3 decimals
    linewidths=0,         # Remove lines between cells
    linecolor='none'      # Ensure no line color
)

plt.title('Corrected P-values from Paired T-tests (Holm-Bonferroni)')
plt.grid(False)
plt.tight_layout()
plt.savefig('Image_Processing_Scripts/significance_matrix.png')
plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/significance_matrix.png', dpi=600)
plt.close()
"""


# # 3. Paired Difference Plots
# # -------------------------
# # For significant pairs, create visualizations of the differences
# for (time1, time2), results in results_all_pairs.items():
#     if results['significant']:
#         # Get paired data
#         paired_data = pivoted_data[[time1, time2]].dropna()
        
#         plt.figure(figsize=(12, 6))
        
#         # Left subplot: Paired line plot
#         plt.subplot(1, 2, 1)
#         for idx in paired_data.index:
#             plt.plot([time1, time2], [paired_data.loc[idx, time1], paired_data.loc[idx, time2]], 'o-', alpha=0.5)
        
#         # Plot mean values
#         mean_vals = [paired_data[time1].mean(), paired_data[time2].mean()]
#         plt.plot([time1, time2], mean_vals, 'r-', linewidth=2, label='Mean')
        
#         plt.title(f'Individual Changes: {time1} vs {time2}')
#         plt.ylabel('Area Difference')
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.legend()
        
#         # Right subplot: Box plot of differences
#         plt.subplot(1, 2, 2)
#         diff = paired_data[time1] - paired_data[time2]
#         sns.boxplot(y=diff)
#         plt.axhline(y=0, color='r', linestyle='--')
#         plt.title(f'Distribution of Differences\np={results["p_corrected"]:.4f}')
#         plt.ylabel(f'{time1} - {time2}')
        
#         plt.tight_layout()
#         plt.savefig(f'paired_diff_{time1}_vs_{time2}.png')
#         plt.show()

# # 4. Additional Summary Plot: Mean differences with confidence intervals
# # ---------------------------------------------------------------------
# # Extract data for all pairs with sufficient data
# pair_data = []
# for _, row in valid_results.iterrows():
#     pair_data.append({
#         'comparison': f"{row['time1']} vs {row['time2']}",
#         'mean_diff': row['mean_diff'],
#         'std_diff': row['std_diff'],
#         'n_pairs': row['n_pairs'],
#         'p_corrected': row['p_corrected'],
#         'significant': row['significant']
#     })

# if pair_data:
#     summary_df = pd.DataFrame(pair_data)
    
#     # Calculate confidence intervals (95%)
#     summary_df['ci_lower'] = summary_df['mean_diff'] - 1.96 * summary_df['std_diff'] / np.sqrt(summary_df['n_pairs'])
#     summary_df['ci_upper'] = summary_df['mean_diff'] + 1.96 * summary_df['std_diff'] / np.sqrt(summary_df['n_pairs'])
    
#     # Sort by mean difference
#     summary_df = summary_df.sort_values('mean_diff')
    
#     # Create forest plot
#     plt.figure(figsize=(10, 6))
    
#     # Plot points and error bars
#     plt.errorbar(
#         summary_df['mean_diff'], 
#         range(len(summary_df)), 
#         xerr=np.array([
#             summary_df['mean_diff'] - summary_df['ci_lower'], 
#             summary_df['ci_upper'] - summary_df['mean_diff']
#         ]),
#         fmt='o', 
#         capsize=5,
#         color=[
#             'red' if sig else 'blue' 
#             for sig in summary_df['significant']
#         ]
#     )
    
#     # Add labels
#     plt.yticks(range(len(summary_df)), summary_df['comparison'])
#     plt.axvline(x=0, color='gray', linestyle='--')
#     plt.title('Mean Differences with 95% Confidence Intervals')
#     plt.xlabel('Mean Difference')
#     plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
#     # Add significance annotation
#     for i, (_, row) in enumerate(summary_df.iterrows()):
#         plt.text(
#             row['ci_upper'] + abs(row['mean_diff'])*0.05, 
#             i, 
#             f"p={row['p_corrected']:.3f}{' *' if row['significant'] else ''}",
#             va='center'
#         )
    
#     plt.tight_layout()
#     plt.savefig('mean_differences_summary.png')
#     plt.show()
















sys.exit()

##############
################
###############################
#################
###########
############
###############
#########old code below #############

# Filter data to include only patients with an 'acute' time point
print('filtering data to include only patients with an acute timepoint')
acute_data = new_df[new_df['timepoint'] == 'acute']
patients_with_acute = acute_data['patient_id'].unique()
print('patients with acute timepoint:', patients_with_acute)
data_filtered = new_df[new_df['patient_id'].isin(patients_with_acute)]

# Pivot data to align each patient’s timepoints as columns
pivoted_data = data_filtered.pivot(index='patient_id', columns='timepoint', values='area_diff')
pivoted_data = pivoted_data.rename_axis(None, axis=1)
#pivoted_data.reset_index(inplace=True)  # Optional, to keep patient_id as a column
print('data pivoted')
print(pivoted_data.head())


# List of timepoints
timepoints = pivoted_data.columns.tolist()

# Initialize dictionary to store results
results = {}
print('running pairwise comparisons')
print(pivoted_data.dtypes)
print(pivoted_data.isnull().sum())

# Define baseline
baseline = 'acute'
# Remove baseline from timepoints
timepoints_without_baseline = [tp for tp in timepoints if tp != baseline]

# Perform pairwise comparisons between 'acute' and each other timepoint
for timepoint in timepoints_without_baseline:
    # Select non-missing pairs for 'acute' and the current timepoint
    paired_data = pivoted_data[[baseline, timepoint]].dropna()

    # If we have enough non-missing pairs, conduct the tests
    if len(paired_data) > 1:  # At least two pairs needed for meaningful test
        # Paired t-test
        try:
            stat_t, p_value_t = ttest_rel(paired_data[baseline], paired_data[timepoint])
        except ValueError as e:
            print(f"T-test error for timepoint {timepoint}: {e}")
            stat_t, p_value_t = np.nan, np.nan

        # Wilcoxon signed-rank test
        try:
            stat_w, p_value_w = wilcoxon(paired_data[baseline], paired_data[timepoint])
        except ValueError as e:
            print(f"Wilcoxon test error for timepoint {timepoint}: {e}")
            stat_w, p_value_w = np.nan, np.nan
    else:
        # If there aren't enough pairs, set values to NaN
        print(f"Not enough pairs for timepoint {timepoint}")
        stat_t, p_value_t, stat_w, p_value_w = np.nan, np.nan, np.nan, np.nan

    # Store results
    results[timepoint] = {
        'statistic_t': stat_t,
        'p_value_t': p_value_t,
        'statistic_w': stat_w,
        'p_value_w': p_value_w
    }

# Convert results to DataFrame
results_df = pd.DataFrame(results).T

# Apply FDR correction on both sets of p-values
results_df['corrected_p_value_t'] = multipletests(results_df['p_value_t'], method='fdr_bh')[1] # fdr_bh
results_df['corrected_p_value_w'] = multipletests(results_df['p_value_w'], method='fdr_bh')[1]

# Reset index to make 'timepoint' a column for easy reference
results_df = results_df.reset_index().rename(columns={'index': 'timepoint'})


# COMPARISON BETWEEN ALL OTHER TIMEPOINTS:
# Initialize dictionary to store all pairwise comparisons
results_all_pairs = {}

# Perform pairwise comparisons among all timepoints excluding the baseline
for timepoint1, timepoint2 in combinations(timepoints_without_baseline, 2):
    paired_data = pivoted_data[[timepoint1, timepoint2]].dropna()

    if len(paired_data) > 1:
        # Paired t-test
        try:
            stat_t, p_value_t = ttest_rel(paired_data[timepoint1], paired_data[timepoint2])
        except ValueError as e:
            print(f"T-test error for pair ({timepoint1}, {timepoint2}): {e}")
            stat_t, p_value_t = np.nan, np.nan

        # Wilcoxon signed-rank test
        try:
            stat_w, p_value_w = wilcoxon(paired_data[timepoint1], paired_data[timepoint2])
        except ValueError as e:
            print(f"Wilcoxon test error for pair ({timepoint1}, {timepoint2}): {e}")
            stat_w, p_value_w = np.nan, np.nan
    else:
        stat_t, p_value_t, stat_w, p_value_w = np.nan, np.nan, np.nan, np.nan

    results_all_pairs[(timepoint1, timepoint2)] = {
        'statistic_t': stat_t,
        'p_value_t': p_value_t,
        'statistic_w': stat_w,
        'p_value_w': p_value_w
    }

# Convert all pairwise comparison results to DataFrame and apply FDR correction
results_all_pairs_df = pd.DataFrame(results_all_pairs).T
results_all_pairs_df.index = pd.MultiIndex.from_tuples(results_all_pairs_df.index, names=["timepoint1", "timepoint2"])
results_all_pairs_df = results_all_pairs_df.reset_index()

results_all_pairs_df['corrected_p_value_t'] = multipletests(results_all_pairs_df['p_value_t'], method='fdr_bh')[1]
results_all_pairs_df['corrected_p_value_w'] = multipletests(results_all_pairs_df['p_value_w'], method='fdr_bh')[1]

#results_all_pairs_df[['timepoint1', 'timepoint2']] = pd.DataFrame(results_all_pairs_df['index'].tolist(), index=results_all_pairs_df.index)
#results_all_pairs_df = results_all_pairs_df.drop(columns=['index'])





# Display results
print(f"Results of pairwise comparisons between {baseline} and other timepoints:")
print(results_df)

print("\nResults of pairwise comparisons among other timepoints:")
print(results_all_pairs_df)





sys.exit()
# Mann-Whitney U Test not valid in this case 
# Perform pairwise Mann-Whitney U Test between 'acute' and each other timepoint
baseline='acute'
for col in pivoted_data.columns:
    if col != baseline:
        # Drop rows where either 'acute' or the comparison timepoint is missing
        paired_data = pivoted_data[[baseline, col]].dropna()
        
        if not paired_data.empty:
            # Perform Mann-Whitney U Test
            stat, p_value = mannwhitneyu(paired_data[baseline], paired_data[col], alternative='greater')
            results[col] = p_value

# Apply FDR correction
timepoints = list(results.keys())
p_values = list(results.values())
adjusted_p_values_fdr = smm.multipletests(p_values, method='fdr_bh')[1]

# Apply Holm-Bonferroni correction
adjusted_p_values_holm = smm.multipletests(p_values, method='holm')[1]

# Display results for both adjustments
print("FDR Adjusted p-values:")
for timepoint, adj_p in zip(timepoints, adjusted_p_values_fdr):
    print(f"Comparison: acute vs {timepoint} -> Adjusted p-value (FDR): {adj_p}")

print("\nHolm-Bonferroni Adjusted p-values:")
for timepoint, adj_p in zip(timepoints, adjusted_p_values_holm):
    print(f"Comparison: acute vs {timepoint} -> Adjusted p-value (Holm-Bonferroni): {adj_p}")

sys.exit()
# Initialize dictionary to store results
results = {}
print('running pairwise comparisons')
# Pairwise comparison for each timepoint with 'acute'
for timepoint in timepoints:
    if timepoint != 'acute':
        # Drop patients with missing values at either timepoint
        valid_pairs = pivoted_data[['acute', timepoint]].dropna()
        
        # Check for sufficient pairs before running the test
        if len(valid_pairs) > 0:
            # Conduct Wilcoxon signed-rank test
            stat, p_value = wilcoxon(valid_pairs['acute'], valid_pairs[timepoint])
            results[f'acute vs {timepoint}'] = {'statistic': stat, 'p_value': p_value}
"""
# Additional pairwise comparisons among non-'acute' timepoints
for tp1, tp2 in combinations([tp for tp in timepoints if tp != 'acute'], 2):
    # Drop patients with missing values at either timepoint
    valid_pairs = pivoted_data[[tp1, tp2]].dropna()
    
    # Check for sufficient pairs before running the test
    if len(valid_pairs) > 0:
        # Conduct Wilcoxon signed-rank test
        stat, p_value = wilcoxon(valid_pairs[tp1], valid_pairs[tp2])
        results[f'{tp1} vs {tp2}'] = {'statistic': stat, 'p_value': p_value}
"""
# Convert results to DataFrame for better readability
results_df = pd.DataFrame(results).T.sort_values(by='p_value')
results_df

print(results_df)


sys.exit()
# below here contains code for mixed effects model - not compatible with current dataset due to sparsity and small sample size
# Fit the model
model = smf.mixedlm("area_diff ~ timepoint", new_df, groups=new_df["patient_id"], re_formula="~1")
result = model.fit()
# Print the summary
print(result.summary())

# Scan by 'first' scan, use this as category for comparison
new_df=new_df.sort_values(by=['patient_id','timepoint'])
new_df['first_scan']=None
print(new_df.head())
# Add a column to the dataframe that indicates the first scan for each patient. first scan is either fast or ultrafast
new_df['first_scan']=new_df.groupby('patient_id')['timepoint'].transform('first')

# Create covariate column specifying the first scan type for each patient
new_df['first_scan_type'] = new_df.apply(lambda row: 'ultra-fast' if row['timepoint'] == 'ultra-fast' 
                                         else 'fast' if row['timepoint'] == 'fast' 
                                         else None, axis=1)

# Forward fill the 'first_scan_type' to propagate the first scan type for each patient across all rows
new_df['first_scan_type'] = new_df.groupby('patient_id')['first_scan_type'].transform('first')

# Create a new column 'scan_type' that specifies the type of scan for each row
new_df['scan_type']=new_df.apply(lambda row: 'first' if row['timepoint']==row['first_scan'] 
                             else 'fast' if row['timepoint'] == 'fast'
                             else 'acute' if row['timepoint'] == 'acute' 
                             else '3mo' if row['timepoint'] == '3mo' 
                             else '6mo' if row['timepoint'] == '6mo' 
                             else '12mo' if row['timepoint'] == '12mo' 
                             else '24mo' if row['timepoint'] == '24mo'
                             else 'other', axis=1)
print(new_df.head())
# set the order of scans
desired_order = ['first', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']   
# Convert 'first_scan_type' to a categorical variable 
new_df['scan_type'] = pd.Categorical(new_df['scan_type'], categories=desired_order, ordered=True)
print(new_df['scan_type'].unique())

# Fit mixed effects model w 'scan_type' and 'first_scan_type' as covariates (fixed effects)
model = smf.mixedlm("area_diff ~ C(scan_type, Treatment(reference='first')) + C(first_scan_type, Treatment(reference='ultra-fast'))", 
                    new_df, 
                    groups=new_df["patient_id"], 
                    re_formula="~1")
result = model.fit()

# Print the summary
print(result.summary())



# plotting

# box plot of area_diff
plt.figure(figsize=(12, 8))
og_order = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
# boxplot of the original data w scan_type on x-axis and area_diff on y-axis
ax=sns.boxplot(x='timepoint', y='area_diff', data=new_df, order=og_order, showfliers=False, whis=5, color="lightblue", width=0.6)

sns.stripplot(x='timepoint', y='area_diff', data=new_df, order=og_order, color="black", alpha=0.5, jitter=True)

# generate predictions from the model
new_df['predicted_area_diff'] = result.fittedvalues
# remove first scan type from the 

# calculate mean predicted values for each scan type group
predicted_means=new_df.groupby('timepoint')['predicted_area_diff'].mean()
print(predicted_means)

# overlay the predicted means on the boxplot
plt.plot(og_order, predicted_means, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=12, label='Predicted Means')

# set labels and title
plt.xlabel('Timepoints', fontsize=12)
plt.ylabel('Area Difference', fontsize=12)
plt.title('Area Difference by Timepoint with Model Predictions', fontsize=14)

plt.xticks(rotation=45)

    

plt.legend()

plt.tight_layout()
plt.savefig('Image_Processing_Scripts/area_diff_boxplot.png')
plt.show()
