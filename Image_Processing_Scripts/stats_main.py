import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import sys
from scipy.stats import mannwhitneyu
import statsmodels.stats.multitest as smm
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests
from itertools import combinations

print('running stats_main.py')
# Load the data (note this data does not contain all timepoints w NaN value if not exist - only contains timepoints w data per original data)
data = pd.read_csv('Image_Processing_Scripts/area_data.csv')
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
