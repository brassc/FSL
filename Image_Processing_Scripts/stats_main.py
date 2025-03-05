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
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
from longitudinal_main import map_timepoint_to_string
from set_publication_style import set_publication_style


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
    Create a scatter plot of area_diff for each timepoint.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Long format dataframe with columns: patient_id, timepoint, area_diff
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
            ax.scatter(x_jittered, tp_data['area_diff'], 
                      color=palette[i], alpha=0.7, s=20, 
                      edgecolor=palette[i], linewidth=0.5)
            
            # Optional: Add mean line
            mean_value = tp_data['area_diff'].mean()
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
    plt.savefig('Image_Processing_Scripts/area_diff_scatter.png')
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/area_diff_scatter.png', dpi=600)
    plt.close()
    return

def create_timepoint_violin(df, timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']):
    """
    Create a violin plot of area_diff for each timepoint with overlaid scatter points.
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
    sns.violinplot(x='timepoint', y='area_diff', data=df_filtered, 
                  palette=palette, inner=None, ax=ax, saturation=0.7, alpha=0.5)
    
    # Add scatter points on top
    sns.stripplot(x='timepoint', y='area_diff', data=df_filtered,
                 palette=palette, jitter=True, size=5, alpha=0.7, ax=ax)
    
    # Add mean markers
    for i, tp in enumerate(timepoints):
        tp_data = df[df['timepoint'] == tp]
        if len(tp_data) > 0:
            mean_value = tp_data['area_diff'].mean()
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
    plt.savefig('Image_Processing_Scripts/area_diff_violin.png')
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/area_diff_violin.png', dpi=600)
    plt.close()
    return

def create_timepoint_boxplot(df, timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']):
    """
    Create a box plot of area_diff for each timepoint with overlaid scatter points.
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
        sns.boxplot(x='timepoint', y='area_diff', data=df_regular,
                  palette=palette, width=0.5, ax=ax, saturation=0.7,
                  showfliers=False)
    
    # Reduce opacity of box elements after creation
    for patch in ax.patches:
        patch.set_alpha(0.5)
    
    
    
    # For small sample sizes (n < 5), plot just the median as a line
    for tp in small_sample_tps:
        tp_data = df[df['timepoint'] == tp]
        tp_index = timepoints.index(tp)
        median_value = tp_data['area_diff'].median()
        
        # Plot median as a horizontal line
        ax.hlines(median_value, tp_index - 0.25, tp_index + 0.25,
                 color='black', linewidth=1.0, linestyle='-',
                 alpha=0.9, zorder=5)
    
    # Add scatter points for all timepoints
    sns.stripplot(x='timepoint', y='area_diff', data=df_filtered,
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
    plt.savefig('Image_Processing_Scripts/area_diff_boxplot_v2.png')
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/area_diff_boxplot_v2.png', dpi=600)
    plt.close()
    return

def data_availability_matrix(data, timepoints):
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
    plt.savefig('Image_Processing_Scripts/data_availability.png')
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/data_availability.png', dpi=600)
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
    Create a box plot of area_diff for each timepoint with overlaid mixed effect model predictions.
    Colors are assigned based on statistical significance.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing patient_id, timepoint, and area_diff columns
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
    chronic_means = chronic_data.groupby('patient_id')['area_diff'].mean().reset_index()
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
        sns.boxplot(x='timepoint', y='area_diff', data=df_regular,
                   palette=palette, width=0.5, ax=ax, saturation=0.7,
                   showfliers=False)
        
        # Reduce opacity of box elements after creation
        for patch in ax.patches:
            patch.set_alpha(0.5)
    
    # For small sample sizes (n < 5), plot just the median as a line
    for tp in small_sample_tps:
        tp_data = df_filtered[df_filtered['timepoint'] == tp]
        tp_index = plot_timepoints.index(tp)
        median_value = tp_data['area_diff'].median()
        
        # Plot median as a horizontal line
        ax.hlines(median_value, tp_index - 0.25, tp_index + 0.25,
                 color='black', linewidth=1.0, linestyle='-',
                 alpha=0.9, zorder=5)
    
    # Add scatter points for all timepoints
    sns.stripplot(x='timepoint', y='area_diff', data=df_filtered,
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
    plt.savefig('Image_Processing_Scripts/area_diff_mixed_effect_boxplot.png', dpi=300)
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/area_diff_mixed_effect_boxplot.png', dpi=600)
    
    return fig




##### MAIN STARTS HERE
#####
#####
#####
if __name__ == '__main__':
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
    #print(data)


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
    #print(new_df.head(10))
    #sys.exit()

    #create_timepoint_scatter(new_df)
    #create_timepoint_violin(new_df)
    #create_timepoint_boxplot(new_df)
    #sys.exit()


    # Check for duplicates
    print("Checking for duplicates:")
    dupes = new_df.duplicated(subset=['patient_id', 'timepoint'], keep=False)
    print(f"Number of duplicate entries: {dupes.sum()}")
    if dupes.sum() > 0:
        print("Sample of duplicates:")
        print(new_df[dupes])# Using all available data

    pivoted_data = new_df.pivot(index='patient_id', columns='timepoint', values='area_diff')
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
    

    # group by patient_id and take the mean of the area_diff
    chronic_means=chronic_data.groupby('patient_id')['area_diff'].mean().reset_index()
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

    # Fit the model: area_diff as outcome, timepoint_category as fixed effect, 
    # patient_id as random effect

    try:
        # Mixed effects model
        model = smf.mixedlm("area_diff ~ timepoint", binned_df, groups=binned_df['patient_id'])
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

        # First, fill in the known p-values from the model summary
        # These are comparisons with the baseline (acute)
        for i, param in enumerate(result.params.index):
            if param.startswith('timepoint'):
                # Extract timepoint name from parameter
                tp = param.split('[T.')[1].rstrip(']')
                if tp in timepoints:
                    j = timepoints.index(tp)
                    p_value = result.pvalues[param]
                    sig_matrix[0, j] = p_value
                    sig_matrix[j, 0] = p_value  # Mirror value
                    print(f"acute vs {tp}: p={p_value:.4f}")

        # Now for the non-baseline comparisons, we need to create new models
        # Approach: Create datasets with only the two timepoints we want to compare
        for i in range(1, n):
            for j in range(i+1, n):
                tp1 = timepoints[i]
                tp2 = timepoints[j]
                
                # Create a subset with just these two timepoints
                subset = binned_df[binned_df['timepoint'].isin([tp1, tp2])].copy()
                
                # Re-categorize with first timepoint as baseline
                subset['timepoint'] = pd.Categorical(
                    subset['timepoint'],
                    categories=[tp1, tp2],
                    ordered=True
                )
                
                # Fit a new model with this subset
                try:
                    sub_model = smf.mixedlm("area_diff ~ timepoint", subset, groups=subset['patient_id'])
                    sub_result = sub_model.fit()
                    
                    # Get p-value for the comparison (should only be one parameter)
                    for param in sub_result.params.index:
                        if param.startswith('timepoint'):
                            p_value = sub_result.pvalues[param]
                            sig_matrix[i, j] = p_value
                            sig_matrix[j, i] = p_value  # Mirror value
                            print(f"{tp1} vs {tp2}: p={p_value:.4f}")
                except Exception as e:
                    print(f"Error comparing {tp1} vs {tp2}: {e}")
                    # Use an alternative approach in case of convergence issues
                    try:
                        # Try with OLS
                        ols_formula = "area_diff ~ C(timepoint) + C(patient_id)"
                        ols_result = smf.ols(ols_formula, subset).fit()
                        
                        # Find the timepoint parameter
                        for param in ols_result.params.index:
                            if 'C(timepoint)' in param and tp2 in param:
                                p_value = ols_result.pvalues[param]
                                sig_matrix[i, j] = p_value
                                sig_matrix[j, i] = p_value  # Mirror value
                                print(f"{tp1} vs {tp2} (OLS): p={p_value:.4f}")
                    except:
                        # Last resort: t-test
                        group1 = subset[subset['timepoint'] == tp1]['area_diff']
                        group2 = subset[subset['timepoint'] == tp2]['area_diff']
                        t_stat, p_value = stats.ttest_ind(group1, group2)
                        sig_matrix[i, j] = p_value
                        sig_matrix[j, i] = p_value  # Mirror value
                        print(f"{tp1} vs {tp2} (t-test): p={p_value:.4f}")

        # Create matrix for visualization
        # Reorder timepoints
        ordered_timepoints = ['ultra-fast', 'fast', 'acute', 'chronic']

        # Reorder the significance matrix
        ordered_sig_matrix = np.zeros((n, n))
        for i, tp1 in enumerate(ordered_timepoints):
            for j, tp2 in enumerate(ordered_timepoints):
                # Get original indices
                orig_i = timepoints.index(tp1)
                orig_j = timepoints.index(tp2)
                # Copy values to new matrix
                ordered_sig_matrix[i, j] = sig_matrix[orig_i, orig_j]

        # Create matrix for visualization
        sig_df = pd.DataFrame(ordered_sig_matrix, index=ordered_timepoints, columns=ordered_timepoints)
        # Set diagonal to NaN for better visualization
        sig_df = sig_df.iloc[::-1, :]
        
        # Mask the anti-diagonal (bottom-left to upper-right)
        n = len(sig_df)
        for i in range(n):
            sig_df.iloc[n-1-i, i] = np.nan  # This sets the anti-diagonal to NaN

        # Create explicit mask for NaN values
        mask = np.isnan(sig_df.values)
        print(sig_df)

        def add_significance_markers(p_df):
            """Add significance markers to a DataFrame of p-values"""
            # Make a copy to avoid modifying the original
            formatted_df = p_df.copy()
            
            # Add markers based on significance levels
            for i in range(len(p_df)):
                for j in range(len(p_df.columns)):
                    if pd.isna(p_df.iloc[i, j]):
                        continue
                        
                    p_value = p_df.iloc[i, j]
                    
                    if p_value < 0.001:
                        formatted_df.iloc[i, j] = f"{p_value:.4f} ***"
                    elif p_value < 0.01:
                        formatted_df.iloc[i, j] = f"{p_value:.4f} **"
                    elif p_value < 0.05:
                        formatted_df.iloc[i, j] = f"{p_value:.4f} *"
                    elif p_value < 0.1:
                        formatted_df.iloc[i, j] = f"{p_value:.4f} †"
                    else:
                        formatted_df.iloc[i, j] = f"{p_value:.4f}"
            
            return formatted_df
        
        annot_df = add_significance_markers(sig_df)
        # # Set diagonal to NaN for better visualization
        # for i in range(len(ordered_timepoints)):
        #     sig_df.iloc[i, i] = np.nan

        # Plot heatmap
        plt.figure(figsize=(10, 10))
        #mask = np.isnan(sig_df)  # Use NaN mask instead of zeros
        cmap = sns.diverging_palette(240, 10, as_cmap=True)

        # Create heatmap without grid lines
        heatmap = sns.heatmap(
            sig_df,
            annot=True,  # Show numbers in cells
            #annot=annot_df,  # Use custom annotations
            cmap=cmap,  # Color map
            mask=mask,  # Mask diagonal values
            vmin=0, vmax=0.5,  # Set color scale range
            center=0.15,  # Center color scale
            fmt='.4f',  # Format as floating point with 4 decimals
            linewidths=0,  # Remove lines between cells
            linecolor='none',  # Ensure no line color
            yticklabels=ordered_timepoints[::-1]  # Reverse y-axis labels
        )

        for i in range(4):  # rows
            for j in range(4):  # columns
                if not pd.isna(sig_df.iloc[i, j]):
                    if sig_df.iloc[i, j] < 0.001:
                        plt.text(
                            j + 0.625,  # x-coordinate (center of column)
                            i + 0.45,  # y-coordinate (center of row)
                            "***",    # annotation text
                            ha='left',  # horizontal alignment
                            va='center',  # vertical alignment
                            #color='red',
                            fontsize=10
                        )
                    elif sig_df.iloc[i, j] < 0.01:
                        plt.text(
                            j + 0.625,  # x-coordinate (center of column)
                            i + 0.45,  # y-coordinate (center of row)
                            "**",    # annotation text
                            ha='left',  # horizontal alignment
                            va='center',  # vertical alignment
                            #color='red',
                            fontsize=10
                        )
                    elif sig_df.iloc[i, j] < 0.05:
                        plt.text(
                            j + 0.625,  # x-coordinate (center of column)
                            i + 0.45,  # y-coordinate (center of row)
                            "*",    # annotation text
                            ha='left',  # horizontal alignment
                            va='center',  # vertical alignment
                            #color='red',
                            fontsize=10
                        )
                    elif sig_df.iloc[i, j] < 0.1:
                        plt.text(
                            j + 0.625,  # x-coordinate (center of column)
                            i + 0.45,  # y-coordinate (center of row)
                            "†",    # annotation text
                            ha='left',  # horizontal alignment
                            va='center',  # vertical alignment
                            #color='red',
                            fontsize=10
                        )
                    
        
        
        plt.title('Pairwise Comparison p-values from Mixed Effects Model')
        plt.grid(False)
        plt.tight_layout()

        # Add better spacing to make room for the legend text
        plt.subplots_adjust(bottom=0.075)  # Increased bottom margin

        # Add legend for significance levels
        plt.figtext(0.25, 0.01, "Significance levels: *** p<0.001, ** p<0.01, * p<0.05, † p<0.1",
                ha='left', fontsize=12, style='italic')
        
        plt.savefig('Image_Processing_Scripts/significance_matrix_mixed_effects.png', dpi=300, bbox_inches='tight')
        plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/significance_matrix_mixed_effects.png', dpi=600, bbox_inches='tight')
        plt.close()



        # pairwise comparisons go here
        # # Create a DataFrame for the Tukey test
        # tukey_data = binned_df[['area_diff', 'timepoint']]

        # # Perform Tukey's HSD test
        # tukey = pairwise_tukeyhsd(endog=tukey_data['area_diff'], 
        #                         groups=tukey_data['timepoint'], 
        #                         alpha=0.05)
        # print("\nTukey's HSD pairwise comparisons:")
        # print(tukey)




    except Exception as e:
        print(f"Error fitting mixed effects model: {e}")
        print("try using OLS with patient_id as dummy variable...")
        formula = "area_diff ~ C(timepoint) + C(patient_id)"
        ols_model = smf.ols(formula, binned_df).fit()
        print(ols_model.summary())


    # Calculate estimated marginal means
    # def calculate_emmeans(result):
    #     """
    #     Calculate Estimated Marginal Means from mixed effects model results
        
    #     Parameters:
    #     - result: Fitted mixed effects model results
        
    #     Returns:
    #     - DataFrame with EMM estimates and confidence intervals
    #     """
    #     # Get parameter names (excluding the first which is the intercept)
    #     param_names = result.model.exog_names[1:]
        
    #     # Initialize results list
    #     emm_results = []
        
    #     # Calculate EMM for each parameter
    #     for param in param_names:
    #         # Get the coefficient for this parameter
    #         coef = result.params[param]
            
    #         # Get standard error
    #         se = result.bse[param]
            
    #         # Calculate confidence interval
    #         t_value = stats.t.ppf(0.975, df=result.df_resid)
    #         ci_lower = coef - t_value * se
    #         ci_upper = coef + t_value * se
            
    #         # Store results
    #         emm_results.append({
    #             'Timepoint': param.split('.')[-1] if '[T.' in param else 'baseline',
    #             'EMM': coef,
    #             'Std Error': se,
    #             'Lower CI (95%)': ci_lower,
    #             'Upper CI (95%)': ci_upper
    #         })
        
    #     return pd.DataFrame(emm_results)
    
    # emmeans = calculate_emmeans(result)
    # print("\nEstimated Marginal Means:")
    # print(emmeans)

    # def visualize_emmeans(result):
    #     """
    #     Visualize Estimated Marginal Means (EMMs) from a mixed effects model
        
    #     Parameters:
    #     - result: Mixed effects model results
        
    #     Returns:
    #     - Figure and accompanying data
    #     """
    #     # Extract parameters related to timepoints
    #     timepoint_params = [param for param in result.params.index if param.startswith('timepoint')]
        
    #     # Prepare EMM data
    #     emm_results = []
    #     baseline_value = result.params['Intercept']  # Get the baseline (intercept) value
        
    #     for param in timepoint_params:
    #         timepoint = param.split('[T.')[-1].rstrip(']')
    #         effect = result.params[param]
    #         std_error = result.bse[param]
    #         p_value = result.pvalues[param]
            
    #         emm_results.append({
    #             'Timepoint': timepoint,
    #             'EMM': baseline_value + effect,  # Adjusted EMM
    #             'Std Error': std_error,
    #             'p-value': p_value
    #         })
        
    #     # Convert to DataFrame
    #     emm_data = pd.DataFrame(emm_results)
        
    #     # Calculate confidence intervals
    #     from scipy import stats
    #     confidence_level = 0.95
    #     degrees_of_freedom = result.df_resid
    #     t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
        
    #     emm_data['Lower CI'] = emm_data['EMM'] - t_value * emm_data['Std Error']
    #     emm_data['Upper CI'] = emm_data['EMM'] + t_value * emm_data['Std Error']
        
    #     # Determine significance
    #     def get_significance(p_value):
    #         if p_value < 0.001:
    #             return '***'
    #         elif p_value < 0.01:
    #             return '**'
    #         elif p_value < 0.05:
    #             return '*'
    #         else:
    #             return 'ns'
        
    #     emm_data['Significance'] = emm_data['p-value'].apply(get_significance)
        
    #     # Create visualization
    #     plt.figure(figsize=(12, 6))
        
    #     # Bar plot with error bars
    #     plt.subplot(1, 2, 1)
    #     x = np.arange(len(emm_data))
    #     plt.bar(x, emm_data['EMM'], yerr=emm_data['Std Error'], 
    #             capsize=5, color='skyblue', edgecolor='black')
    #     plt.title('Estimated Marginal Means')
    #     plt.xlabel('Timepoint')
    #     plt.ylabel('Area Difference')
    #     plt.xticks(x, emm_data['Timepoint'], rotation=45)
        
    #     # Add significance annotations
    #     for i, row in enumerate(emm_data.itertuples()):
    #         plt.text(i, row.EMM + row._3, row.Significance, 
    #                 horizontalalignment='center')
        
    #     # Detailed results table
    #     plt.subplot(1, 2, 2)
    #     plt.axis('off')
    #     table_data = [
    #         ['Timepoint', 'EMM', 'Lower CI', 'Upper CI', 'p-value', 'Sig']
    #     ]
    #     for _, row in emm_data.iterrows():
    #         table_data.append([
    #             row['Timepoint'], 
    #             f"{row['EMM']:.2f}", 
    #             f"{row['Lower CI']:.2f}", 
    #             f"{row['Upper CI']:.2f}", 
    #             f"{row['p-value']:.4f}", 
    #             row['Significance']
    #         ])
        
    #     table = plt.table(cellText=table_data, 
    #                     loc='center', 
    #                     cellLoc='center', 
    #                     colWidths=[0.15]*6)
    #     table.auto_set_font_size(False)
    #     table.set_fontsize(9)
    #     table.scale(1.2, 1.5)
    #     plt.title('EMM Statistical Details')
        
    #     plt.tight_layout()
        
    #     return plt.gcf(), emm_data

    # # Example usage:
    # fig, emm_df = visualize_emmeans(result)
    # plt.show()



    




    





    ### Visualisations

    # Raw data visualisations:
    ## create_timepoint_scatter(new_df)
    ## create_timepoint_violin(new_df)
    #create_timepoint_boxplot(new_df)

    # pairwise test visualisations:
    #data_availability_matrix(pivoted_data, timepoints)
    #significance_matrix_ttest(valid_results, timepoints, 'significance_matrix.png')
    #significance_matrix_wilcoxon(valid_wilcoxon_results, timepoints, 'significance_matrix_wilcoxon.png')
    #create_forest_plot(valid_wilcoxon_results, 'mean_differences_summary.png')

    # Mixed effect model visualisations:


    #mixed_effect_boxplot(new_df, result, timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo'], chronic_timepoints=['3mo', '6mo', '12mo', '24mo'])
    
    











































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
    #print('patients with acute timepoint:', patients_with_acute)
    data_filtered = new_df[new_df['patient_id'].isin(patients_with_acute)]

    # Pivot data to align each patient’s timepoints as columns
    pivoted_data = data_filtered.pivot(index='patient_id', columns='timepoint', values='area_diff')
    pivoted_data = pivoted_data.rename_axis(None, axis=1)
    #pivoted_data.reset_index(inplace=True)  # Optional, to keep patient_id as a column
    print('data pivoted')



    # List of timepoints
    timepoints = pivoted_data.columns.tolist()

    # Initialize dictionary to store results
    results = {}
    print('running pairwise comparisons')
    # print(pivoted_data.dtypes)
    # print(pivoted_data.isnull().sum())

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
