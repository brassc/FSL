import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import seaborn as sns
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from scipy.stats import mannwhitneyu
import statsmodels.stats.multitest as smm
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
from longitudinal_main import map_timepoint_to_string
from set_publication_style import set_publication_style
import pickle as pkl
import rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import PackageNotInstalledError

# pandas2ri conversion will be used via context manager where needed

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



set_publication_style()

# Function to perform Wilcoxon signed-rank test between two time points
def wilcoxon_signed_rank_test(data, time1, time2):
    # Get data for both time points, excluding rows with missing values in either
    paired_data = data[[time1, time2]].dropna()
    
    if len(paired_data) < 2:
        return {
            'comparison': f"{time1} vs {time2}",
            'time1': time1,
            'time2': time2,
            'n_pairs': len(paired_data),
            'median_diff': np.nan,
            'w_statistic': np.nan,
            'p_value': np.nan,
            'std_diff': np.nan,
            'sufficient_data': False
        }
    
    # Calculate the differences between the two time points
    diff = paired_data[time1] - paired_data[time2]
    
    # Perform Wilcoxon signed-rank test
    w_stat, p_val = stats.wilcoxon(paired_data[time1], paired_data[time2])
    
    # Calculate the median difference
    median_diff = np.median(diff)

    # Calculate the standard deviation of the differences
    std_diff = np.std(diff, ddof=1)  # ddof=1 for sample standard deviation
    
    return {
        'comparison': f"{time1} vs {time2}",
        'time1': time1,
        'time2': time2,
        'n_pairs': len(paired_data),
        'median_diff': median_diff,
        'w_statistic': w_stat,
        'p_value': p_val,
        'std_diff': std_diff,
        'sufficient_data': True
    }

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


# Visualisation/plot functions

def create_timepoint_scatter(df, timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']):
    """
    Create a scatter plot of h_diff for each timepoint.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Long format dataframe with columns: patient_id, timepoint, h_diff
    timepoints : list
        List of timepoints in the desired order
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import matplotlib.cm as cm
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set color palette
    palette = sns.color_palette("plasma", len(timepoints)) #cm.tab10(np.linspace(0.9, 0.1, len(timepoints)))
    
    # Create x positions for each timepoint (equally spaced)
    positions = np.arange(len(timepoints))
    
    # Add a slight jitter to avoid overlapping points
    jitter_width = 0.2
    
    # Plot each timepoint
    for i, tp in enumerate(timepoints):
        # Extract data for this timepoint
        tp_data = df[df['timepoint'] == tp]
        
        if len(tp_data) > 0:
            # Add jitter to x positions
            x_jittered = np.random.normal(positions[i], jitter_width, size=len(tp_data))
            
            # Plot scatter points
            ax.scatter(x_jittered, tp_data['h_diff'], 
                      color=palette[i], alpha=0.7, s=20, 
                      edgecolor=palette[i], linewidth=0.5)
            
            # Optional: Add mean line
            mean_value = tp_data['h_diff'].mean()
            ax.hlines(mean_value, positions[i]-0.3, positions[i]+0.3, 
                     color=palette[i], linewidth=2, linestyle='-')
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Area Difference', fontsize=12)
    ax.set_title('Area Difference by Timepoint', fontsize=14, fontweight='bold')
    
    # Set x-ticks at the positions, with timepoint labels
    ax.set_xticks(positions)
    ax.set_xticklabels(timepoints, rotation=45)
    
    # Add grid for y-axis only
    ax.grid(True, axis='y', linestyle='-', alpha=0.3)
    
    # Show count of patients per timepoint
    for i, tp in enumerate(timepoints):
        count = len(df[df['timepoint'] == tp])
        if count > 0:
            ax.text(positions[i], ax.get_ylim()[0] * 2.0, f"n={count}", 
                   ha='center', va='bottom', fontsize=10)
    ax.xaxis.set_label_coords(0.5, -0.25)  # Move x-axis label down
    plt.tight_layout()
    plt.savefig('Image_Processing_Scripts/h_diff_scatter.png')
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/h_diff_scatter.png', dpi=600)
    plt.close()
    return

def create_timepoint_violin(df, timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']):
    """
    Create a violin plot of h_diff for each timepoint with overlaid scatter points.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import matplotlib.cm as cm
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    # Set color palette
    palette = sns.color_palette("plasma", len(timepoints))
    
    # Filter the dataframe to include only timepoints in the specified list
    df_filtered = df[df['timepoint'].isin(timepoints)].copy()
    
    # Ensure timepoints are in the correct order
    df_filtered['timepoint'] = pd.Categorical(df_filtered['timepoint'], 
                                             categories=timepoints, 
                                             ordered=True)
    
    # Create violin plot
    sns.violinplot(x='timepoint', y='h_diff', data=df_filtered, 
                  palette=palette, inner=None, ax=ax, saturation=0.7, alpha=0.5)
    
    # Add scatter points on top
    sns.stripplot(x='timepoint', y='h_diff', data=df_filtered,
                 palette=palette, jitter=True, size=5, alpha=0.7, ax=ax)
    
    # Add mean markers
    for i, tp in enumerate(timepoints):
        tp_data = df[df['timepoint'] == tp]
        if len(tp_data) > 0:
            mean_value = tp_data['h_diff'].mean()
            ax.hlines(mean_value, i-0.3, i+0.3,
                     color=palette[i], linewidth=2, linestyle='-')
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Area Difference', fontsize=12)
    ax.set_title('Area Difference by Timepoint', fontsize=14, fontweight='bold')
    
    # Add grid for y-axis only
    ax.grid(True, axis='y', linestyle='-', alpha=0.3)
    
    # Show count of patients per timepoint
    for i, tp in enumerate(timepoints):
        count = len(df[df['timepoint'] == tp])
        if count > 0:
            ax.text(i, ax.get_ylim()[0] * 1.3, f"n={count}",
                   ha='center', va='bottom', fontsize=10)
    
    ax.xaxis.set_label_coords(0.5, -0.125) # Move x-axis label down
    plt.tight_layout()
    plt.savefig('Image_Processing_Scripts/h_diff_violin.png')
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/h_diff_violin.png', dpi=600)
    plt.close()
    return

def create_timepoint_boxplot(df, timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']):
    """
    Create a box plot of h_diff for each timepoint with overlaid scatter points.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.cm as cm
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set color palette
    palette = sns.color_palette("plasma", len(timepoints))
    
    # Filter the dataframe to include only timepoints in the specified list
    df_filtered = df[df['timepoint'].isin(timepoints)].copy()
    
    # Ensure timepoints are in the correct order
    df_filtered['timepoint'] = pd.Categorical(df_filtered['timepoint'],
                                             categories=timepoints,
                                             ordered=True)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Create a copy of the data for non-small sample sizes
    df_regular = df_filtered.copy()
    df_small = df_filtered.copy()
    
    # Create lists to track which timepoints have small sample sizes
    small_sample_tps = []
    regular_sample_tps = []
    
    for tp in timepoints:
        tp_data = df[df['timepoint'] == tp]
        if len(tp_data) < 5:
            small_sample_tps.append(tp)
        else:
            regular_sample_tps.append(tp)
    
    # Create a mask for each dataset
    if small_sample_tps:
        df_small = df_small[df_small['timepoint'].isin(small_sample_tps)]
    else:
        df_small = df_small[df_small['timepoint'] == 'none_placeholder']
        
    if regular_sample_tps:
        df_regular = df_regular[df_regular['timepoint'].isin(regular_sample_tps)]
    else:
        df_regular = df_regular[df_regular['timepoint'] == 'none_placeholder']
    
    # Plot regular boxplots for n >= 5
    if not df_regular.empty:
        sns.boxplot(x='timepoint', y='h_diff', data=df_regular,
                  palette=palette, width=0.5, ax=ax, saturation=0.7,
                  showfliers=False)
    
    # Reduce opacity of box elements after creation
    for patch in ax.patches:
        patch.set_alpha(0.5)
    
    
    
    # For small sample sizes (n < 5), plot just the median as a line
    for tp in small_sample_tps:
        tp_data = df[df['timepoint'] == tp]
        tp_index = timepoints.index(tp)
        median_value = tp_data['h_diff'].median()
        
        # Plot median as a horizontal line
        ax.hlines(median_value, tp_index - 0.25, tp_index + 0.25,
                 color='black', linewidth=1.0, linestyle='-',
                 alpha=0.9, zorder=5)
    
    # Add scatter points for all timepoints
    sns.stripplot(x='timepoint', y='h_diff', data=df_filtered,
                 palette=palette, jitter=True, size=6, alpha=0.8, ax=ax)
        
    # Set labels and title
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Herniation Area [mm²]', fontsize=12)
    ax.set_title('Herniation Area by Timepoint', fontsize=14, fontweight='bold')
    
    # Add grid for y-axis only
    ax.grid(True, axis='y', linestyle='-', alpha=0.3)
    
    # Show count of patients per timepoint
    for i, tp in enumerate(timepoints):
        count = len(df[df['timepoint'] == tp])
        if count > 0:
            ax.text(i, ax.get_ylim()[0] * 1.5, f"n={count}",
                   ha='center', va='bottom', fontsize=10)
    
    ax.xaxis.set_label_coords(0.5, -0.125) # Move x-axis label down
    plt.tight_layout()
    plt.savefig('Image_Processing_Scripts/h_diff_boxplot_v2.png')
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/h_diff_boxplot_v2.png', dpi=600)
    plt.close()
    return

def data_availability_matrix(data, timepoints, filename='data_availability.png'):
    """
    Create a data availability matrix for the given timepoints.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Pivoted data with patient IDs as index and timepoints as columns
    timepoints : list
        List of timepoints to include in the matrix
        
    Returns:
    --------
    availability_matrix : pandas.DataFrame
        Matrix showing the number of patients with data for each pair of timepoints
    """
    # Create an empty matrix with timepoints as index and columns
    availability_matrix = pd.DataFrame(index=timepoints, columns=timepoints, dtype=float)

    # Fill the matrix with counts
    for time1 in timepoints:
        for time2 in timepoints:
            if time1 == time2:
                # Diagonal: number of non-missing values for this time point
                availability_matrix.loc[time1, time2] = data[time1].notna().sum()
            else:
                # Off-diagonal: number of patients with data for both time points
                common_data = data[[time1, time2]].dropna()
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
    plt.savefig(f"Image_Processing_Scripts/{filename}")
    plt.savefig(f"../Thesis/phd-thesis-template-2.4/Chapter5/Figs/{filename}", dpi=600)
    plt.close()


    return

def significance_matrix_ttest(valid_results, timepoints, filename):
    # Create a matrix of corrected p-values
    significance_matrix = pd.DataFrame(index=timepoints, columns=timepoints, dtype=float)
    significance_matrix.fillna(1.0, inplace=True)  # Default p-value = 1 (not significant)

    # Fill in the values from our results
    for _, row in valid_results.iterrows():
        significance_matrix.loc[row['time1'], row['time2']] = row['p_holm']
        significance_matrix.loc[row['time2'], row['time1']] = row['p_holm']  # Mirror since it's symmetric

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
    plt.savefig(f'Image_Processing_Scripts/{filename}')
    plt.savefig(f'../Thesis/phd-thesis-template-2.4/Chapter5/Figs/{filename}', dpi=600)
    plt.close()

    return 

def significance_matrix_wilcoxon(valid_wilcoxon_results, timepoints, filename):
        # Create a matrix of corrected p-values
    significance_matrix_wilcoxon = pd.DataFrame(index=timepoints, columns=timepoints, dtype=float)
    significance_matrix_wilcoxon.fillna(1.0, inplace=True)  # Default p-value = 1 (not significant)

    # Fill in the values from results
    for _, row in valid_wilcoxon_results.iterrows():
        significance_matrix_wilcoxon.loc[row['time1'], row['time2']] = row['p_value']
        significance_matrix_wilcoxon.loc[row['time2'], row['time1']] = row['p_value']  # Mirror since it's symmetric

    # Set diagonal to NaN for better visualization
    for time in timepoints:
        significance_matrix_wilcoxon.loc[time, time] = np.nan

    # Visualize the significance matrix
    plt.figure(figsize=(10, 8))
    mask = np.isnan(significance_matrix_wilcoxon)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # Create heatmap without grid lines
    heatmap = sns.heatmap(
        significance_matrix_wilcoxon,
        annot=True,           # Show numbers in cells
        cmap=cmap,            # Color map
        mask=mask,            # Mask diagonal values
        vmin=0, vmax=1,     # Set color scale range
        center=0.3,          # Center color scale
        fmt='.3f',            # Format as floating point with 3 decimals
        linewidths=0,         # Remove lines between cells
        linecolor='none'      # Ensure no line color
    )

    plt.title('Raw P-values from Wilcoxon Signed-Rank Test')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('Image_Processing_Scripts/significance_matrix_wilcoxon_uncorrected.png')
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/significance_matrix_wilcoxon_uncorrected.png', dpi=600)
    plt.close()

    return

def create_forest_plot(valid_wilcoxon_results, filename):

    # 4. Additional Summary Plot: Mean differences with confidence intervals
    # ---------------------------------------------------------------------
    # Extract data for all pairs with sufficient data for wilcoxon results
    pair_data = []
    for _, row in valid_wilcoxon_results.iterrows():
        pair_data.append({
            'comparison': f"{row['time1']} vs {row['time2']}",
            'mean_diff': row['median_diff'],
            'n_pairs': row['n_pairs'],
            'p_corrected': row['p_corrected'],
            'significant': row['significant'],
            'std_diff': row['std_diff']
        })
    if pair_data:
        summary_df = pd.DataFrame(pair_data)
        print(summary_df)
        # Calculate confidence intervals (95%)
        summary_df['ci_lower'] = summary_df['mean_diff'] - 1.96 * summary_df['std_diff'] / np.sqrt(summary_df['n_pairs'])
        summary_df['ci_upper'] = summary_df['mean_diff'] + 1.96 * summary_df['std_diff'] / np.sqrt(summary_df['n_pairs'])
        # Sort by mean difference
        #summary_df = summary_df.sort_values('mean_diff')
        # Check that the number of colors matches the number of rows in summary_df
        colors = ['red' if sig else 'blue' for sig in summary_df['significant']]
        
        # Create forest plot
        plt.figure(figsize=(10, 6))
        plt.grid(True, axis='x', linestyle='-', alpha=0.3)
        plt.grid(False, axis='y')

        n_comparisons = len(summary_df)
        
        # Plot points and error bars
        for i in range(len(summary_df)):
            plt.errorbar(
                summary_df.iloc[i]['mean_diff'],
                n_comparisons - 1 - i,  # Reverse the y position
                xerr=[[summary_df.iloc[i]['mean_diff'] - summary_df.iloc[i]['ci_lower']],
                    [summary_df.iloc[i]['ci_upper'] - summary_df.iloc[i]['mean_diff']]],
                fmt='o',
                capsize=5,
                color=colors[i], 
                #markersize=8,
                #elinewidth=2,
                #capthick=2
            )
        
        # Labels
        plt.yticks(range(n_comparisons), list(reversed(summary_df['comparison'])))
        #plt.axvline(x=0, color='gray', linestyle='-', linewidth=1.5, alpha=0.7)
        plt.title('Mean Differences in Herniation Area Between Timepoint Pairs \nwith 95% Confidence Intervals, Corrected p-values (FDR) and Number of each Pair')
        plt.xlabel('Mean Area Difference [mm²]')
        
        
        # Add significance annotation (with reversed positions)
        max_x=max([x['ci_upper'] for _, x in summary_df.iterrows()]) * 1.1
        for i in range(len(summary_df)):
            plt.text(
                max_x,  # Align all p-values to same x position
                #i,
                n_comparisons - 1 - i,  # Reverse the y position
                f"p={summary_df.iloc[i]['p_corrected']:.3f}{' *' if summary_df.iloc[i]['significant'] else ''}",
                va='center',
                fontsize=11,
                ha='left'
            )
        
        # Add number of pair annotation (with reversed positions)
        max_x2=max([x['ci_upper'] for _, x in summary_df.iterrows()]) * 1.3
        for i in range(len(summary_df)):
            plt.text(
                max_x2,  # Align all p-values to same x position
                #i,
                n_comparisons - 1 - i,  # Reverse the y position
                f"n={summary_df.iloc[i]['n_pairs']}{' *' if summary_df.iloc[i]['significant'] else ''}",
                va='center',
                fontsize=11,
                ha='left'
            )


            
        
        plt.tight_layout()
        plt.savefig(f'Image_Processing_Scripts/{filename}')
        plt.savefig(f'../Thesis/phd-thesis-template-2.4/Chapter5/Figs/{filename}', dpi=600)
        plt.close()

    return

def mixed_effect_boxplot(df, result, timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo'], 
                   chronic_timepoints=['3mo', '6mo', '12mo', '24mo']):
    """
    Create a box plot of h_diff for each timepoint with overlaid mixed effect model predictions.
    Colors are assigned based on statistical significance.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing patient_id, timepoint, and h_diff columns
    result : statsmodels MixedLMResults
        The fitted mixed effects model result object
    timepoints : list
        List of timepoints to include in the plot, in desired order
    chronic_timepoints : list
        List of timepoints to be categorized as 'chronic'
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D
    from matplotlib.colors import to_rgba
    from matplotlib.cm import coolwarm
    
    # Create a copy of the dataframe
    new_df = df.copy()
    new_df['patient_id'] = new_df['patient_id'].astype(str)
    
    # Create a new combined dataframe for plotting with 'chronic' timepoint
    # Create new timepoint category
    def categorize_timepoint(timepoint):
        if timepoint in chronic_timepoints:
            return 'chronic'
        else:
            return timepoint
    
    new_df['timepoint_category'] = new_df['timepoint'].apply(categorize_timepoint)
    
    # Get chronic data and calculate means per patient
    chronic_data = new_df[new_df['timepoint_category'] == 'chronic']
    chronic_means = chronic_data.groupby('patient_id')['h_diff'].mean().reset_index()
    chronic_means['timepoint'] = 'chronic'
    
    # Remove original chronic timepoints and add back the means
    non_chronic_data = new_df[new_df['timepoint_category'] != 'chronic']
    
    # Create a modified timepoints list for plotting
    # First, find all chronic timepoints in the original list
    chronic_indices = [i for i, tp in enumerate(timepoints) if tp in chronic_timepoints]
    
    # Skip if no chronic timepoints
    if not chronic_indices:
        df_filtered = new_df.copy()
        plot_timepoints = timepoints
    else:
        # Insert 'chronic' at the position of the first chronic timepoint
        plot_timepoints = timepoints.copy()
        first_chronic_index = min(chronic_indices)
        
        # Create a modified list with chronic
        plot_timepoints = [tp for tp in timepoints if tp not in chronic_timepoints]
        plot_timepoints.insert(first_chronic_index, 'chronic')
        
        # Create the combined dataframe for plotting
        combined_data = pd.concat([non_chronic_data, chronic_means], ignore_index=True)
        df_filtered = combined_data.copy()
        
    # Create figure and axis with adjusted size for publication
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set color palette - adjust for the modified timepoints
    palette = sns.color_palette("plasma", len(plot_timepoints))
    
    # Ensure timepoints are in the correct order
    if 'chronic' in plot_timepoints:
        df_filtered['timepoint'] = pd.Categorical(df_filtered['timepoint'],
                                               categories=plot_timepoints,
                                               ordered=True)
    else:
        df_filtered['timepoint'] = pd.Categorical(df_filtered['timepoint'],
                                               categories=timepoints,
                                               ordered=True)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Create a copy of the data for non-small sample sizes
    df_regular = df_filtered.copy()
    df_small = df_filtered.copy()
    
    # Create lists to track which timepoints have small sample sizes
    small_sample_tps = []
    regular_sample_tps = []
    
    for tp in plot_timepoints:
        tp_data = df_filtered[df_filtered['timepoint'] == tp]
        if len(tp_data) < 5:
            small_sample_tps.append(tp)
        else:
            regular_sample_tps.append(tp)
    
    # Create a mask for each dataset
    if small_sample_tps:
        df_small = df_small[df_small['timepoint'].isin(small_sample_tps)]
    else:
        df_small = df_small[df_small['timepoint'] == 'none_placeholder']
    
    if regular_sample_tps:
        df_regular = df_regular[df_regular['timepoint'].isin(regular_sample_tps)]
    else:
        df_regular = df_regular[df_regular['timepoint'] == 'none_placeholder']
    
    # Plot regular boxplots for n >= 5
    if not df_regular.empty:
        sns.boxplot(x='timepoint', y='h_diff', data=df_regular,
                   palette=palette, width=0.5, ax=ax, saturation=0.7,
                   showfliers=False)
        
        # Reduce opacity of box elements after creation
        for patch in ax.patches:
            patch.set_alpha(0.5)
    
    # For small sample sizes (n < 5), plot just the median as a line
    for tp in small_sample_tps:
        tp_data = df_filtered[df_filtered['timepoint'] == tp]
        tp_index = plot_timepoints.index(tp)
        median_value = tp_data['h_diff'].median()
        
        # Plot median as a horizontal line
        ax.hlines(median_value, tp_index - 0.25, tp_index + 0.25,
                 color='black', linewidth=1.0, linestyle='-',
                 alpha=0.9, zorder=5)
    
    # Add scatter points for all timepoints
    sns.stripplot(x='timepoint', y='h_diff', data=df_filtered,
                 palette=palette, jitter=True, size=6, alpha=0.8, ax=ax)
    
    # Extract coefficients from the statsmodels results object
    fe_params = result.fe_params
    intercept = fe_params['Intercept']
    
    # Determine the reference category and extract coefficients
    reference_category = None
    predictions = {}
    coefficient_categories = []
    
    # Extract all categories from the parameter names
    for param in fe_params.index:
        if param != 'Intercept' and '[T.' in param:
            category = param.split('[T.')[-1].rstrip(']')
            coefficient_categories.append(category)
    
    # Determine reference category (not present in coefficients)
    all_categories = ['acute', 'ultra-fast', 'fast', 'chronic']
    for category in all_categories:
        if category not in coefficient_categories:
            reference_category = category
            break
    
    if reference_category is None:
        reference_category = 'acute'  # Default if we can't determine
    
    # Create predictions dictionary
    predictions = {reference_category: intercept}  # Reference category = intercept only
    
    # Add predictions for other categories
    for param, value in fe_params.items():
        if param != 'Intercept' and '[T.' in param:
            category = param.split('[T.')[-1].rstrip(']')
            predictions[category] = intercept + value
    
    # Use coolwarm colormap for significance-based coloring
    model_order = ['acute', 'ultra-fast', 'fast', 'chronic']
    
    # Create colors based on significance
    custom_colors = {}
    sig_levels = {}  # Store significance levels for use in the legend
    
    for category in model_order:
        if category == reference_category:  # Baseline
            custom_colors[category] = to_rgba(coolwarm(0.3))  # Balance point (neutral)
            sig_levels[category] = "baseline"
        else:
            # Check statistical significance
            param_name = f'timepoint[T.{category}]'
            if param_name in result.pvalues:
                p_value = result.pvalues[param_name]
                
                # Assign significance level
                if p_value < 0.001:
                    sig_levels[category] = "***"
                elif p_value < 0.01:
                    sig_levels[category] = "**"
                elif p_value < 0.05:
                    sig_levels[category] = "*"
                elif p_value < 0.1:
                    sig_levels[category] = "†"
                else:
                    sig_levels[category] = "n.s."
                
                # Assign color based on significance
                if p_value < 0.05:  # Significant
                    custom_colors[category] = to_rgba(coolwarm(0.0))  # Blue for significant
                else:  # Not significant
                    custom_colors[category] = to_rgba(coolwarm(1.0))  # Red for not significant
            else:
                custom_colors[category] = to_rgba(coolwarm(0.3))  # Default to neutral
                sig_levels[category] = "n.s."
    
    # Draw the predicted values for each category in the plot using the new colors
    for i, tp in enumerate(plot_timepoints):
        if tp in predictions:
            pred_value = predictions[tp]
            ax.scatter(i, pred_value, color=custom_colors.get(tp, 'black'), 
                      marker='o', s=120, zorder=10, 
                      edgecolor='black', linewidth=1.5)
    
    # Create a more professional legend
    legend_elements = []
    
    # Create legend title with better formatting
    title = "Mixed Effect Model Predictions"
    
        # Model predictions in legend with p-values
    for category in model_order:
        if category in predictions:
            # Format the label with p-values
            if category == reference_category:
                label = f"{category.capitalize()} reference"
            else:
                param_name = f'timepoint[T.{category}]'
                if param_name in result.pvalues:
                    p_value = result.pvalues[param_name]
                    # Format p-value with appropriate precision
                    if p_value < 0.001:
                        p_text = "(p < 0.001) ***"
                    elif p_value < 0.01:
                        p_text = f"(p < {p_value:.3f}) **"
                    else:
                        p_text = f"(p < {p_value:.3f}) †"
                    label = f"{category.capitalize()} {p_text}"
                else:
                    label = f"{category.capitalize()}"
            
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=custom_colors.get(category, 'black'),
                      markeredgecolor='black',
                      markersize=10, label=label)
            )
    
    # Create the legend with improved formatting
    if legend_elements:
        legend = ax.legend(handles=legend_elements, title=title, 
                          loc='upper right', framealpha=0.8,
                          fontsize=9, title_fontsize=10)
        
        # Get figure dimensions and calculate position for significance notation
        fig_width = fig.get_figwidth()
        fig_height = fig.get_figheight()
        
        # Position in the upper right, below where the legend is expected to be
        # Using figure coordinates (0-1 range)
        plt.figtext(0.975, 0.7, "Blue: Significant. Red: Not Significant. \nSignificance levels: \n*** p<0.001, ** p<0.01, * p<0.05, † p<0.1",
               ha='right', fontsize=8, style='italic')
        

    # Add significance notations under the x-axis
    for i, tp in enumerate(plot_timepoints):
        if tp != reference_category and tp in sig_levels:
            sig_marker = sig_levels.get(tp, "")
            if sig_marker not in ["baseline", "n.s."]:
                ax.text(i, -1500, sig_marker, 
                       ha='center', va='top', fontsize=12, color='black')
    
    # Set labels and title
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Herniation Area [mm²]', fontsize=12)
    ax.set_title('Herniation Area by Timepoint with Mixed Effect Model', fontsize=14, fontweight='bold')
    
    # Add grid for y-axis only
    ax.grid(True, axis='y', linestyle='-', alpha=0.3)
    
    # Show count of patients per timepoint
    for i, tp in enumerate(plot_timepoints):
        count = len(df_filtered[df_filtered['timepoint'] == tp])
        if count > 0:
            ax.text(i, ax.get_ylim()[0] * 1.5, f"n={count}",
                   ha='center', va='bottom', fontsize=10)
    
    
    
    ax.xaxis.set_label_coords(0.5, -0.125)  # Move x-axis label down
    plt.tight_layout()
    plt.savefig('Image_Processing_Scripts/h_diff_mixed_effect_boxplot.png', dpi=300)
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/h_diff_mixed_effect_boxplot.png', dpi=600)
    
    return fig

# returns sig_df, mask
def emmeans_significance_matrix(py_pairs):
    timepoints= ['ultra-fast', 'fast', 'acute', 'chronic']
    n=len(timepoints)
    sig_matrix=np.ones((n, n))

    # Fill matrix w p_values from py_pairs
    for index, row in py_pairs.iterrows():
        # parse the contrast to get the two groups being compared
        contrast = row['contrast']

        if ' - (' in contrast:
            # Case like "acute - (ultra-fast)"
            parts = contrast.split(' - (')
            group1 = parts[0].strip()
            group2 = parts[1].strip(')')
        elif ') - ' in contrast:
            # Case like "(ultra-fast) - fast"
            parts = contrast.split(') - ')
            group1 = parts[0].strip('(')
            group2 = parts[1].strip()
        elif ' - ' in contrast:
            # Case like "acute - fast"
            parts = contrast.split(' - ')
            group1 = parts[0].strip()
            group2 = parts[1].strip()
        else:
            print(f"Unexpected contrast format: {contrast}")
            continue

        # Debug print to see what we're extracting
        #print(f"Extracted: '{group1}' and '{group2}' from '{contrast}'")

        

        # find the indices of the groups
        i = timepoints.index(group1)
        j = timepoints.index(group2)

        # Store the p-value in both positions (for symmetry)

        
        
        sig_matrix[i,j]=row['p.value']
        sig_matrix[j,i]=row['p.value']

    
    # Create DataFrame for visualization
    sig_df = pd.DataFrame(sig_matrix, index=timepoints, columns=timepoints)
    
    # mask diagonal values
    for i in range(n):
        sig_df.iloc[i, i] = np.nan
    
            
    # invert for better visualisation
    #sig_df = sig_df.iloc[::-1, :]
    print(sig_df)
    mask = np.isnan(sig_df.values)

    return sig_df, mask
def plot_emmeans_sig_mat_h(sig_df, mask):
    # Plot heatmap
    ordered_timepoints= ['ultra-fast', 'fast', 'acute', 'chronic']
    plt.figure(figsize=(10, 10))
    # cmap = sns.diverging_palette(240, 10, as_cmap=True)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#9dd1ef", "#ff9797"]) # Pale blue for significant, pale red for non-significant

    # Create heatmap
    heatmap = sns.heatmap(
        sig_df,
        annot=True,  # Show numbers in cells
        annot_kws={"size": 16},  # Annotation font size
        cmap=cmap,   # Color map
        mask=mask,   # Mask diagonal values
        # vmin=0, vmax=0.5,  # Set color scale range
        # center=0.15,  # Center color scale
        vmin=0, vmax=0.1, # Binary threshold at 0.05
        center=0.05, # Center at significance threshold
        fmt='.4f',   # Format as floating point with 4 decimals
        linewidths=0,  # Remove lines between cells
        linecolor='none',  # Ensure no line color
        yticklabels=ordered_timepoints,  # Reverse y-axis labels
        # cbar_kws={"label": "p-value"} # Add colorbar label
        cbar=False, 
        square=True  # Make cells square-shaped
    )
    # Set axis to be square
    heatmap.set_aspect('equal')

    # Make x and y axis labels bigger
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # # Make colorbar tick labels bigger
    # cbar = heatmap.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=14)

    # Add binary legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#9dd1ef', label='Significant (p < 0.05)'),
        Patch(facecolor='#ff9797', label='Not significant (p ≥ 0.05)')
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left',
               fontsize=14, frameon=True)

    # Add significance markers
    for i in range(4):  # rows
        for j in range(4):  # columns
            if not pd.isna(sig_df.iloc[i, j]):
                if sig_df.iloc[i, j] < 0.001:
                    plt.text(
                        j + 0.625,  # x-coordinate
                        i + 0.45,   # y-coordinate
                        "***",      # annotation text
                        ha='left',  # horizontal alignment
                        va='center', # vertical alignment
                        fontsize=10
                    )
                elif sig_df.iloc[i, j] < 0.01:
                    plt.text(
                        j + 0.665, i + 0.45, "**", ha='left', va='center', fontsize=10
                    )
                elif sig_df.iloc[i, j] < 0.05:
                    plt.text(
                        j + 0.665, i + 0.45, "*", ha='left', va='center', fontsize=10
                    )
                # elif sig_df.iloc[i, j] < 0.1:
                #     plt.text(
                #         j + 0.65, i + 0.45, "†", ha='left', va='center', fontsize=10
                #     )

    plt.title('Pairwise comparison of p-values from mixed effects model (emmeans) for ellipse $h$ parameter')
    plt.grid(False)
    plt.tight_layout()

    # Add better spacing for legend
    plt.subplots_adjust(bottom=0.075)

    # Add legend for significance levels
    # plt.figtext(0.25, 0.01, "Significance levels: ** p<0.01, * p<0.05, † p<0.1",
    #         ha='left', fontsize=12, style='italic') #*** p<0.001, 

    plt.savefig('significance_matrix_mixed_effects_h.png', dpi=300, bbox_inches='tight')
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/significance_matrix_mixed_effects_h.png', dpi=600, bbox_inches='tight')
    plt.close()



##### MAIN STARTS HERE
#####
#####
#####
if __name__ == '__main__':
    print('running stats_main.py')
    
    # Load the data (note this data does not contain all timepoints w NaN value if not exist - only contains timepoints w data per original data)
    # Load the data from batch2_ellipse_data.pkl and ellipse_data.pkl
    batch1_data=pkl.load(open('Image_Processing_Scripts/ellipse_data.pkl', 'rb'))
    batch2_data=pkl.load(open('Image_Processing_Scripts/batch2_ellipse_data.pkl', 'rb'))
    batch2_data['timepoint'] = batch2_data['timepoint'].apply(map_timepoint_to_string) # convert to string category
    # Combine the two batches
    data = pd.concat([batch1_data, batch2_data], ignore_index=True)
    # retain only the columns we need h_param_def, h_param_ref, patient_id, timepoint
    data = data[['patient_id', 'timepoint', 'h_param_def', 'h_param_ref']]
    # do h_param_def - h_param_ref to get h_diff
    data['h_diff'] = data['h_param_def'] - data['h_param_ref']
    # drop h_param_def and h_param_ref
    data = data.drop(columns=['h_param_def', 'h_param_ref'], axis=1)

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
    #print(data)

    new_df = data.copy()
    new_df['patient_id'] = new_df['patient_id'].astype(str)
    new_df['timepoint'] = new_df['timepoint'].astype(str)
    new_df['h_diff'] = new_df['h_diff'].astype(float)

    # Drop rows where h_diff is NaN
    new_df = new_df.dropna(subset=['h_diff']) # there should be no NaN values in h_diff, but just in case
    print('rows dropped')
    # Ensure categorical values are categorical
    print('ensuring categorical values are categorical')
    new_df['timepoint']=pd.Categorical(new_df['timepoint'], categories=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo'])
    
    # Now get new ranges for categories based on date since injury
    sep_df =new_df.copy()
    print(sep_df.head(100))
    
    days_data=pd.read_csv('DTI_Processing_Scripts/patient_scanner_data_with_timepoints.csv')
    days_data=days_data.drop(columns=['Cohort','Site','Model','Scan_date'], axis=1)
    pd.set_option('display.max_rows', None)  # Show all rows
    #print(days_data.head(100))

    
    # remove rows that have no value in timepoint
    days_data=days_data.dropna(subset=['timepoint'])
    #print(days_data.head(100))
    
    # sort values by patient ID and timepoint in order ultra-fast, fast, acute, 3mo, 6mo, 12mo, 24mo
    timepoint_order=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
    days_data['timepoint']=pd.Categorical(days_data['timepoint'], categories=timepoint_order)
    days_data=days_data.sort_values(by=['patient_id', 'timepoint']).reset_index(drop=True)
    print(days_data.head(100))
    #if patient_id and timepoint combination not in sep_df, pop days_data row
    # remove duplicates
    days_data=days_data.drop_duplicates(subset=['patient_id', 'timepoint']).reset_index(drop=True)
    print(days_data.head(100))

    # remove duplicates from days_data
    #days_data=days_data.drop_duplicates(subset=['patient_id', 'timepoint'], keep='first')
    #print(days_data.head(100))
    # Create a set of patient_id and timepoint combinations from sep_df
    # Fix data types to ensure consistent comparison
    sep_df['patient_id'] = sep_df['patient_id'].astype(str)
    days_data['patient_id'] = days_data['patient_id'].astype(str)

    # Normalize timepoint strings
    sep_df['timepoint'] = sep_df['timepoint'].str.strip()
    days_data['timepoint'] = days_data['timepoint'].str.strip()

    # Filter days_data to keep only rows with matching combinations in sep_df
    valid_pairs = set(zip(sep_df['patient_id'], sep_df['timepoint']))
    filtered_days_data = days_data[
        days_data.apply(lambda row: (str(row['patient_id']), row['timepoint']) in valid_pairs, axis=1)
    ]

    # Merge with the filtered dataframe
    sep_df = sep_df.merge(
        filtered_days_data[['patient_id', 'timepoint', 'Days_since_injury']],
        on=['patient_id', 'timepoint'],
        how='left'
    )
    print(sep_df)

    # Redo ranges

    recategorised_df = sep_df.copy()
    ranges = [(0,2), (2,8), (8, 42), (42, 179), (179, 278), (278, 540), (540, 500000)]
    labels = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
    recategorised_df['timepoint'] = pd.cut(recategorised_df['Days_since_injury'], bins=[0, 2, 8, 42, 179, 278, 540, 500000], labels=labels)
    
    #print rows where recategorised_df is different from sep_df
    print(recategorised_df[recategorised_df['timepoint'] != sep_df['timepoint']])
    # drop duplicate patient_id timepoint combinations
    recategorised_df=recategorised_df.drop_duplicates(subset=['patient_id', 'timepoint'])

    #create_timepoint_boxplot_recategorised(recategorised_df)
    

    new_df=recategorised_df.drop(columns='Days_since_injury', axis=1)




    # Check for duplicates
    print("Checking for duplicates:")
    dupes = new_df.duplicated(subset=['patient_id', 'timepoint'], keep=False)
    print(f"Number of duplicate entries: {dupes.sum()}")
    if dupes.sum() > 0:
        print("Sample of duplicates:")
        print(new_df[dupes])# Using all available data

    pivoted_data = new_df.pivot(index='patient_id', columns='timepoint', values='h_diff')
    pivoted_data = pivoted_data.rename_axis(None, axis=1)

    
    #print(pivoted_data.head(10))
    


    """
    # Add deformation status
    pivoted_data_with_deformation = pivoted_data.copy()
    pivoted_data_with_deformation['has_deformation'] = True
    non_def_patients = ['12519', '19575', '19981', '21221', '2ZFz639', '6shy992', '9GfT823']
    # Update the deformation status for specified patients
    for patient_id in non_def_patients:
        if patient_id in pivoted_data_with_deformation.index:
            pivoted_data_with_deformation.loc[patient_id, 'has_deformation'] = False

    # Verify the result
    print("Deformation status added:")
    print(pivoted_data_with_deformation[['has_deformation']])  # Display the first 10 rows

    # Check that the specified patients have False status
    for patient_id in non_def_patients:
        if patient_id in pivoted_data_with_deformation.index:
            print(f"Patient {patient_id} has_deformation: {pivoted_data_with_deformation.loc[patient_id, 'has_deformation']}")
    """

    # # combine timepoints - combine 6mo, 12mo and 24mo
    # # Combine 6mo, 12mo, and 24mo into a single timepoint
    # pivoted_data['12-24mo'] = pivoted_data[['12mo', '24mo']].mean(axis=1, skipna=True)
    # pivoted_data = pivoted_data.drop(columns=['12mo', '24mo'])

    ## Paired testing
    """
    # DO FOR WHOLE DATASET FIRST
    # List of all timepoints
    timepoints = pivoted_data.columns.tolist()

    # Initialise dictionary to store results
    results_all_pairs = {}

    # Perform pairwise comparisons among all timepoints

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

    # Run Wilcoxon signed-rank tests
    wilcoxon_results = []
    for time1, time2 in pairs:
        wilcoxon_result = wilcoxon_signed_rank_test(pivoted_data, time1, time2)
        wilcoxon_results.append(wilcoxon_result)
    # Convert results to DataFrame
    wilcoxon_results_df = pd.DataFrame(wilcoxon_results)



    # Filter to show only pairs with sufficient data
    valid_results = results_df[results_df['sufficient_data'] == True].copy()
    valid_wilcoxon_results = wilcoxon_results_df[wilcoxon_results_df['sufficient_data'] == True].copy()

    if len(valid_results) > 0:
        # Apply Holm-Bonferroni correction to p-values
        if len(valid_results) > 1:  # Only apply if there are multiple tests
            p_values = valid_results['p_value'].values
            rejected_holm, p_corrected_holm, _, _ = multipletests(p_values, alpha=0.05, method='holm')
            valid_results['p_holm'] = p_corrected_holm
            valid_results['significant_holm'] = rejected_holm

            # Apply FDR (Benjamini-Hochberg) correction
            rejected_fdr, p_corrected_fdr, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
            valid_results['p_fdr'] = p_corrected_fdr
            valid_results['significant_fdr'] = rejected_fdr

        else:
            # If only one test, no correction needed
            valid_results['p_holm'] = valid_results['p_value']
            valid_results['significant_holm'] = valid_results['p_value'] < 0.05
            valid_results['p_fdr'] = valid_results['p_value']
            valid_results['significant_fdr'] = valid_results['p_value'] < 0.05
        
        # Sort by corrected p-value
        valid_results = valid_results.sort_values('p_value')
        
        
        # Print results for all testing methods
        print("\nPaired t-test Results (All p-values):")
        print(valid_results[['comparison', 'n_pairs', 'mean_diff', 't_statistic', 'p_value', 'p_holm', 'significant_holm', 'p_fdr', 'significant_fdr']])
        

        
        # Store results in a dictionary for later use
        results_all_pairs = {}
        for _, row in valid_results.iterrows():
            key = (row['time1'], row['time2'])
            results_all_pairs[key] = {
                'n_pairs': row['n_pairs'],
                'mean_diff': row['mean_diff'],
                't_statistic': row['t_statistic'],
                'p_value': row['p_value'],
                'p_holm': row['p_holm'],
                'significant_holm': row['significant_holm'],
                'p_fdr': row['p_fdr'],
                'significant_fdr': row['significant_fdr']
            }
    else:
        print("No pairs have sufficient data for paired t-test.")
        results_all_pairs = {}

    results_all_pairs_wilcoxon = {}
    if len(valid_wilcoxon_results) > 0:
        if len(valid_wilcoxon_results) > 1:
            # Apply FDR correction 
            p_values_wilcoxon = valid_wilcoxon_results['p_value'].values
            rejected_wilcoxon, p_corrected_wilcoxon, _, _ = multipletests(p_values_wilcoxon, alpha=0.05, method='fdr_bh')
            valid_wilcoxon_results['p_corrected'] = p_corrected_wilcoxon
            valid_wilcoxon_results['significant'] = rejected_wilcoxon

        else:
            # If only one test, no correction needed
            valid_wilcoxon_results['p_corrected'] = valid_wilcoxon_results['p_value']
            valid_wilcoxon_results['significant'] = valid_wilcoxon_results['p_value'] < 0.05

        # Sort by corrected p-value
        #valid_wilcoxon_results = valid_wilcoxon_results.sort_values('p_value')

        # Print results
        print("\nWilcoxon Signed-Rank Test Results:")
        print(valid_wilcoxon_results[['comparison', 'n_pairs', 'median_diff', 'w_statistic', 'p_value', 'p_corrected', 'significant']])
        
        # Store results in dictionary for later use
        results_all_pairs_wilcoxon = {}

        for _, row in valid_wilcoxon_results.iterrows():
            key = (row['time1'], row['time2'])
            results_all_pairs_wilcoxon[key] = {
                'n_pairs': row['n_pairs'],
                'median_diff': row['median_diff'],
                'w_statistic': row['w_statistic'],
                'p_value': row['p_value'],
                'p_corrected': row['p_corrected'],
                'significant': row['significant']
            }
    else:
        print("No pairs have sufficient data for Wilcoxon signed-rank test.")
        results_all_pairs_wilcoxon = {}

    """
        
    ### Deformed patients only
    """
    # TEST FOR DEFORMED PATIENTS ONLY

    # Filter the data to keep only patients with deformation
    deformed_patients_data = pivoted_data_with_deformation[pivoted_data_with_deformation['has_deformation'] == True]

    # Remove the deformation status column to keep only the timepoint columns
    deformed_patients_data = deformed_patients_data.drop('has_deformation', axis=1)

    # Print information about the filtered dataset
    print(f"Original dataset: {pivoted_data.shape[0]} patients")
    print(f"Filtered dataset (deformed only): {deformed_patients_data.shape[0]} patients")
    print(f"Patients excluded: {pivoted_data.shape[0] - deformed_patients_data.shape[0]}")

    # List of all timepoints (excluding the deformation column)
    timepoints = [col for col in pivoted_data.columns.tolist() if col != 'has_deformation']

    # Initialize dictionary to store results
    results_all_pairs_deformed = {}

    # Perform pairwise comparisons using the filtered data
    ttest_results_deformed = []
    for time1, time2 in combinations(timepoints, 2):
        ttest_result = paired_ttest(deformed_patients_data, time1, time2)
        ttest_results_deformed.append(ttest_result)

    # Convert results to DataFrame
    results_df_deformed = pd.DataFrame(ttest_results_deformed)

    # Filter to show only pairs with sufficient data
    valid_results_deformed = results_df_deformed[results_df_deformed['sufficient_data'] == True].copy()

    if len(valid_results_deformed) > 0:
        # Apply Holm-Bonferroni correction to p-values
        if len(valid_results_deformed) > 1:  # Only apply if there are multiple tests
            p_values = valid_results_deformed['p_value'].values
            rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
            valid_results_deformed['p_corrected'] = p_corrected
            valid_results_deformed['significant'] = rejected
        else:
            # If only one test, no correction needed
            valid_results_deformed['p_corrected'] = valid_results_deformed['p_value']
            valid_results_deformed['significant'] = valid_results_deformed['p_value'] < 0.05
        
        # Sort by corrected p-value
        valid_results_deformed = valid_results_deformed.sort_values('p_corrected')
        
        # Print results
        print("\nPaired t-test Results for DEFORMED PATIENTS ONLY with Holm-Bonferroni correction:")
        print(valid_results_deformed[['comparison', 'n_pairs', 'mean_diff', 't_statistic', 'p_value', 'p_corrected', 'significant']])
        
        # Store results in a dictionary for later use
        results_all_pairs_deformed = {}
        for _, row in valid_results_deformed.iterrows():
            key = (row['time1'], row['time2'])
            results_all_pairs_deformed[key] = {
                'n_pairs': row['n_pairs'],
                'mean_diff': row['mean_diff'],
                't_statistic': row['t_statistic'],
                'p_value': row['p_value'],
                'p_corrected': row['p_corrected'],
                'significant': row['significant']
            }
    else:
        print("No pairs have sufficient data for paired t-test in the deformed patients subset.")
        results_all_pairs_deformed = {}

    # Compare the results between all patients and only deformed patients
    if len(results_all_pairs) > 0 and len(results_all_pairs_deformed) > 0:
        print("\nComparison of results between all patients and only deformed patients:")
        
        # Create a summary table
        comparison_rows = []
        
        # Get all unique pairs from both result sets
        all_pairs = set(results_all_pairs.keys()).union(set(results_all_pairs_deformed.keys()))
        
        for pair in all_pairs:
            time1, time2 = pair
            comparison = f"{time1} vs {time2}"
            
            # Get results for all patients
            if pair in results_all_pairs:
                all_patients_result = results_all_pairs[pair]
                all_p = all_patients_result['p_fdr']
                all_sig = all_patients_result['significant_fdr']
            else:
                all_p = float('nan')
                all_sig = False
            
            # Get results for deformed patients only
            if pair in results_all_pairs_deformed:
                deformed_result = results_all_pairs_deformed[pair]
                deformed_p = deformed_result['p_corrected']
                deformed_sig = deformed_result['significant']
            else:
                deformed_p = float('nan')
                deformed_sig = False
            
            # Add to comparison rows
            comparison_rows.append({
                'Comparison': comparison,
                'All Patients p-value': all_p,
                'All Patients Significant': all_sig,
                'Deformed Only p-value': deformed_p,
                'Deformed Only Significant': deformed_sig,
                'Change in Significance': all_sig != deformed_sig
            })
        
        # Create DataFrame and sort by change in significance
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df = comparison_df.sort_values(['Change in Significance', 'All Patients p-value'], ascending=[False, True])
        
        # Display the comparison
        pd.set_option('display.max_rows', None)  # Show all rows
        print(comparison_df)
        
        # Highlight the differences
        differences = comparison_df[comparison_df['Change in Significance']]
        if not differences.empty:
            print("\nTimepoint comparisons with CHANGES in significance after excluding non-deformed patients:")
            print(differences[['Comparison', 'All Patients p-value', 'All Patients Significant', 
                            'Deformed Only p-value', 'Deformed Only Significant']])
        else:
            print("\nNo changes in significance were found after excluding non-deformed patients.")
    """
    
    ## Mixed effects model
    df = new_df.copy()
    df['patient_id'] = df['patient_id'].astype(str)

    # Create new timepoint category
    def categorize_timepoint(timepoint):
        if timepoint in ['3mo', '6mo', '12mo', '24mo']:
            return 'chronic'
        else:
            return timepoint
        
    df['timepoint_category'] = df['timepoint'].apply(categorize_timepoint)
    # if patient has multiple timepoints, take the mean
    chronic_data=df[df['timepoint_category']=='chronic']
    

    # group by patient_id and take the mean of the h_diff
    chronic_means=chronic_data.groupby('patient_id')['h_diff'].mean().reset_index()
    chronic_means['timepoint'] = 'chronic'

    # remove original chronic timepoints and add back the means
    non_chronic_data=df[df['timepoint_category']!='chronic']
    combined_data=pd.concat([non_chronic_data, chronic_means], ignore_index=True)

    # sort by patient_id then timepoint
    order=['ultra-fast', 'fast', 'acute', 'chronic']
    combined_data['timepoint_order'] = combined_data['timepoint'].apply(lambda x: order.index(x))
    
    combined_data = combined_data.sort_values(by=['patient_id', 'timepoint_order'])
    combined_data = combined_data.drop(['timepoint_category', 'timepoint_order'], axis=1) # remove sorting column
    
    
    binned_df = combined_data.copy()

    # Optional: Check how many patients have data for each category
    patient_categories = binned_df.groupby('patient_id')['timepoint'].apply(list)
    print("\nCategories per patient:")
    print(patient_categories.head(10))

    # Convert timepoint_category to a categorical variable with ultra-fast as the reference
    binned_df['timepoint'] = pd.Categorical(
    binned_df['timepoint'], 
    categories=['acute', 'ultra-fast', 'fast', 'chronic'], # acute as first category - use this as baseline
    ordered=True
    )

    #print(binned_df)
    # pivot the data for plotting
    binned_df_pivot = binned_df.pivot(index='patient_id', columns='timepoint', values='h_diff')
    #print(binned_df_pivot)
    # reorder columns to match the order of the categories
    binned_df_pivot = binned_df_pivot[['ultra-fast', 'fast', 'acute', 'chronic']]
    # plot data availability matrix
    
    

    # Fit the model: h_diff as outcome, timepoint_category as fixed effect, 
    # patient_id as random effect

    #try:
        # Mixed effects model
    model = smf.mixedlm("h_diff ~ timepoint", binned_df, groups=binned_df['patient_id'])
    result = model.fit()
    print("mixed effect model summary:")
    print(result.summary())

    # Extract and display the key results
    print("\nFixed Effects Parameters:")
    print(result.fe_params)
    
    print("\nRandom Effects Parameters:")
    print(result.cov_re)

    # pairwise comparisons go here
    # Define all timepoints
    timepoints = ['acute', 'ultra-fast', 'fast', 'chronic']
    n = len(timepoints)
    sig_matrix = np.ones((n, n))  # Initialize with 1s (diagonal)

    print("\nPairwise Comparisons:")

    # Tukey test - doesnt account for random effects structure of the model
    """
    # Now perform Tukey's HSD test on the data
    # Note: Tukey's HSD is performed on the raw data, not on the model -
    #This means it doesn't account for the random effects structure of your model.
    tukey = pairwise_tukeyhsd(
        endog=binned_df['h_diff'],     # The dependent variable
        groups=binned_df['timepoint'],    # The grouping variable
        alpha=0.05                        # The significance level
    )

    # Print the Tukey HSD results
    print(tukey)

    ## You can also visualize the results
    fig = tukey.plot_simultaneous()
    """

    # EMMEANS - estimates of marginal means
    # Get the estimated marginal means


    # Use context manager for pandas to R conversion
    with (ro.default_converter + pandas2ri.converter).context():
        # Convert your data to R
        r_df = ro.conversion.py2rpy(binned_df)

        # Create and fit the model in R
        ro.globalenv['data'] = r_df
        formula = "h_diff ~ timepoint + (1 | patient_id)"
        r_model = lme4.lmer(formula, data=ro.globalenv['data'])

        # Get the estimated marginal means
        r_emmeans = emmeans.emmeans(r_model, specs="timepoint")
        r_pairs = emmeans.contrast(r_emmeans, method="pairwise", adjust="tukey")
        # tukey adjustment controls family wise error rate, accounting for variance structure in mixed model including random effects.

        # Convert the results back to Python
        py_pairs = ro.conversion.rpy2py(base.as_data_frame(r_pairs))

    print(py_pairs)
    

    
    
    





    




    





    ### Visualisations

    # Raw data visualisations:
    ## create_timepoint_scatter(new_df)
    ## create_timepoint_violin(new_df)
    #create_timepoint_boxplot(new_df)

    # pairwise test visualisations:
    #data_availability_matrix(pivoted_data, timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo'], filename='data_availability.png')
    #significance_matrix_ttest(valid_results, timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo'], filename='significance_matrix.png')
    #significance_matrix_wilcoxon(valid_wilcoxon_results, timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo'], filename='significance_matrix_wilcoxon.png')
    #create_forest_plot(valid_wilcoxon_results, 'mean_differences_summary.png')

    # Mixed effect model visualisations:
    #data_availability_matrix(binned_df_pivot, order, filename='data_availability_matrix_binned.png')

    # Plot the significance matrix
    sig_df, mask = emmeans_significance_matrix(py_pairs)
    plot_emmeans_sig_mat_h(sig_df, mask)  

    #mixed_effect_boxplot(new_df, result, timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo'], chronic_timepoints=['3mo', '6mo', '12mo', '24mo'])
   
