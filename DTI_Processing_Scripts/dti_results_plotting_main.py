import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.interpolate import CubicSpline
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
from matplotlib.lines import Line2D
import matplotlib as mpl
import statsmodels.formula.api as smf
import seaborn as sns
import sys
import os
import re
import scipy.stats as stats
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


def print_fixed_effects_summary_precise(result, precision=6):
    """
    Print Fixed Effects (OLS) model summary with specified precision
    Parameters:
    result: OLS regression results from statsmodels
    precision: number of decimal places (default=6)
    """
    print("                            OLS Regression Results                            ")
    print("==============================================================================")
    print(f"Dep. Variable:                FA_diff   R-squared:                       {result.rsquared:.{precision}f}")
    print(f"Model:                            OLS   Adj. R-squared:                  {result.rsquared_adj:.{precision}f}")
    print("Method:                 Least Squares   F-statistic:                     {:.{precision}f}".format(result.fvalue, precision=precision))
    print("Date:                Thu, 29 May 2025   Prob (F-statistic):             {:.{precision}f}".format(result.f_pvalue, precision=precision))
    print("Time:                        17:55:47   Log-Likelihood:                 {:.{precision}f}".format(result.llf, precision=precision))
    print(f"No. Observations:                  {result.nobs:<2.0f}   AIC:                            {result.aic:.{precision}f}")
    print(f"Df Residuals:                      {result.df_resid:<2.0f}   BIC:                            {result.bic:.{precision}f}")
    print(f"Df Model:                           {result.df_model:<2.0f}                                         ")
    print("Covariance Type:              cluster                                         ")
    print("===========================================================================================")
    print("                              coef    std err          z      P>|z|      [0.025      0.975]")
    print("-------------------------------------------------------------------------------------------")
    
    # Format each parameter row
    for param_name in result.params.index:
        coef = result.params[param_name]
        std_err = result.bse[param_name] 
        z_val = result.tvalues[param_name]
        p_val = result.pvalues[param_name]
        conf_int = result.conf_int().loc[param_name]
        
        print(f"{param_name:<26} {coef:>10.{precision}f}      {std_err:.{precision}f}     {z_val:>6.{precision}f}      {p_val:.{precision}f}       {conf_int[0]:.{precision}f}       {conf_int[1]:.{precision}f}")
    
    print("==============================================================================")
    
    # Additional statistics (using default precision since these may not be available)
    try:
        print(f"Omnibus:                       {result.omnibus[0]:.{precision}f}   Durbin-Watson:                   {result.durbin_watson:.{precision}f}")
        print(f"Prob(Omnibus):                  {result.omnibus[1]:.{precision}f}   Jarque-Bera (JB):               {result.jarque_bera[0]:.{precision}f}")
        print(f"Skew:                           {result.skew:.{precision}f}   Prob(JB):                     {result.jarque_bera[1]:.2e}")
        print(f"Kurtosis:                       {result.kurtosis:.{precision}f}   Cond. No.                         {result.condition_number:.2f}")
    except:
        pass
    
    print("==============================================================================")
    print("Notes:")
    print("[1] Standard Errors are robust to cluster correlation (cluster)")
    print("Fixed Effects Parameters:")
    print(result.params.round(precision))



def print_lme_summary_precise(result, precision=5):
    """
    Print Mixed Linear Model summary with specified precision
    Parameters:
    result: MixedLMResults object from statsmodels
    precision: number of decimal places (default=8)
    """
    print("            Mixed Linear Model Regression Results")
    print("==============================================================")
    print(f"Model:            MixedLM Dependent Variable: {result.model.endog_names}")
    
    # Calculate n_groups from group_labels
    n_groups = len(result.model.group_labels)
    group_sizes = [len(group) for group in result.model.group_labels]
    
    print(f"No. Observations: {result.nobs:<8.0f}  Method:             REML            ")
    print(f"No. Groups:       {n_groups:<8.0f}  Scale:              {result.scale:.{precision}f}      ")
    print(f"Min. group size:  {min(group_sizes):<8.0f}   Log-Likelihood:     {result.llf:.{precision}f}     ")
    print(f"Max. group size:  {max(group_sizes):<8.0f}   Converged:          Yes             ")
    print(f"Mean group size:  {result.nobs/n_groups:.1f}                                         ")
    print("----------------------------------------------------------------")
    print("              Coef.      Std.Err.        z        P>|z|      [0.025      0.975]")
    print("----------------------------------------------------------------")
    
    # Format each parameter row (excluding Group Var)
    for param_name in result.params.index:
        if param_name != 'Group Var':
            coef = result.params[param_name]
            std_err = result.bse[param_name]
            z_val = result.tvalues[param_name]
            p_val = result.pvalues[param_name]
            conf_int = result.conf_int().loc[param_name]
            
            print(f"{param_name:<12} {coef:>10.{precision}f} {std_err:>10.{precision}f} {z_val:>8.{precision}f} {p_val:>8.{precision}f} {conf_int[0]:>10.{precision}f} {conf_int[1]:>10.{precision}f}")

    # Group variance
    print(f"Group Var    {result.cov_re.iloc[0,0]:.{precision}f}      {result.bse_re.iloc[0]:.{precision}f}                                  ")
    print("==============================================================")
    
    # Parameters with more precision
    print(result.params.round(precision))
    
    # Covariance matrix with more precision  
    print(result.cov_re.round(precision))




def create_timepoint_boxplot_LME_dti_old(df, parameter, result, timepoints=['ultra-fast', 'fast', 'acute', '3-6mo', '12-24mo']):
    """
    Create a box plot with anterior and posterior data represented as different colored points,
    but with a single boxplot per timepoint combining both anterior and posterior data.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.cm as cm
    import os

    set_publication_style()  # Assuming this function exists in your environment

    fig, ax = plt.subplots(figsize=(12, 6))

    palette = sns.color_palette("viridis", len(timepoints))

    # Column Names
    anterior_column = f"{parameter}_anterior_diff"
    posterior_column = f"{parameter}_posterior_diff"

    # We need to reshape the data to have a single "value" column
    # First, filter to only the timepoints we want
    df_filtered = df[df['timepoint'].isin(timepoints)].copy()
    
    # Create a melted dataframe for plotting
    # This reshapes the data to have a single value column with a region identifier
    melted_data = pd.melt(
        df_filtered,
        id_vars=['timepoint'], 
        value_vars=[anterior_column, posterior_column],
        var_name='region_col',
        value_name='diff_value'
    )
    
    # Create a cleaner region column
    melted_data['region'] = melted_data['region_col'].apply(
        lambda x: 'Anterior' if 'anterior' in x else 'Posterior'
    )
    
    # Ensure timepoints are in the correct order
    melted_data['timepoint'] = pd.Categorical(melted_data['timepoint'],
                                             categories=timepoints,
                                             ordered=True)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Create lists to track which timepoints have small sample sizes
    small_sample_tps = []
    regular_sample_tps = []
    
    for tp in timepoints:
        # Count unique subjects in this timepoint (assuming one row per subject per timepoint)
        tp_data = melted_data[melted_data['timepoint'] == tp]
        tp_count = len(df[df['timepoint'] == tp])  # Original count from df
        if tp_count < 5:
            small_sample_tps.append(tp)
        else:
            regular_sample_tps.append(tp)
    
    # Create a copy for regular samples
    melted_regular = melted_data[melted_data['timepoint'].isin(regular_sample_tps)].copy()
    
    estimate_points = []

    for tp in timepoints:
        # Get the fixed effect estimate (relative to intercept)
        if tp == "acute":
            est = result.params["Intercept"]
            se = result.bse["Intercept"]
        else:
            coef_name = f"timepoint[T.{tp}]"
            est = result.params["Intercept"] + result.params.get(coef_name, 0)
            se = (result.bse["Intercept"] ** 2 + result.bse.get(coef_name, 0) ** 2) ** 0.5
        
        estimate_points.append((tp, est, se))


    ### PLOTTING
    # Plot regular boxplots for n >= 5
    if not melted_regular.empty:
        sns.boxplot(x='timepoint', y='diff_value', data=melted_regular,
                  palette=palette, width=0.6, ax=ax, saturation=0.7,
                  showfliers=False)
    
    # For small sample sizes (n < 5), plot just the median as a line
    for tp in small_sample_tps:
        tp_data = melted_data[melted_data['timepoint'] == tp]
        if not tp_data.empty and not tp_data['diff_value'].isna().all():
            tp_index = timepoints.index(tp)
            median_value = tp_data['diff_value'].median()
            
            # Plot median as a horizontal line
            ax.hlines(median_value, tp_index - 0.3, tp_index + 0.3,
                    color='black', linewidth=1.5, linestyle='-',
                    alpha=0.9, zorder=5)
            
    # PLOT LME RESULTS OVER THE TOP
    # for i, (tp, est, se) in enumerate(estimate_points):
    #     ax.errorbar(
    #         i, est, yerr=se, fmt='D', color='black', capsize=4, label='LME Estimate' if i == 0 else "",
    #         markersize=6, zorder=6  # Appear above everything
    #     )

    # # Add LME intercept line
    # intercept = result.params["Intercept"]
    # ax.axhline(y=intercept, color='black', linestyle='--', linewidth=1.2, alpha=0.7, label="LME Intercept")

    # Extract LME predictions

    
    x_positions = np.arange(len(timepoints))
    y_estimates = [est for _, est, _ in estimate_points]
    y_errors = [se for _, _, se in estimate_points]

    # Calculate confidence intervals
    ci_lower = np.array(y_estimates) - 1.96 * np.array(y_errors)
    ci_upper = np.array(y_estimates) + 1.96 * np.array(y_errors)

    # Extend to box edges (left edge of first box to right edge of last box)
    box_width = 0.6  # Should match your boxplot width
    x_extended = np.linspace(-box_width/2, len(timepoints)-1+box_width/2, 100)


    # Create step function instead of linear interpolation
    x_step = []
    y_step = []
    ci_lower_step = []
    ci_upper_step = []

    box_width = 0.6
    for i, (x, y, ci_l, ci_u) in enumerate(zip(x_positions, y_estimates, ci_lower, ci_upper)):
        if i == 0:
            # First timepoint: calculate half the distance to next timepoint
            half_dist_to_next = (x_positions[1] - x) / 2
            x_step.extend([x - half_dist_to_next, x + half_dist_to_next])
            y_step.extend([y, y])
            ci_lower_step.extend([ci_l, ci_l])
            ci_upper_step.extend([ci_u, ci_u])
        else:
            midpoint = (x_positions[i-1] + x) / 2
            
            # Extend previous timepoint to midpoint
            x_step.append(midpoint)
            y_step.append(y_estimates[i-1])
            ci_lower_step.append(ci_lower[i-1])
            ci_upper_step.append(ci_upper[i-1])
            
            # Step up/down at midpoint
            x_step.append(midpoint)
            y_step.append(y)
            ci_lower_step.append(ci_l)
            ci_upper_step.append(ci_u)
            
            # Continue to next midpoint or end
            if i == len(x_positions) - 1:
                # Last timepoint: extend by half the distance from previous
                half_dist_from_prev = (x - x_positions[i-1]) / 2
                x_step.append(x + half_dist_from_prev)
                y_step.append(y)
                ci_lower_step.append(ci_l)
                ci_upper_step.append(ci_u)


    # Plot step function
    ax.fill_between(x_step, ci_lower_step, ci_upper_step, 
                    alpha=0.1, color='#440154', label='LME 95% CI', step='post')
    ax.plot(x_step, y_step, '-', color='#440154', linewidth=2.5, 
            label='LME Estimate', zorder=10, drawstyle='steps-post')
    ax.plot(x_positions, y_estimates, 'o', markerfacecolor='#6B5082', 
        markersize=10, markeredgecolor='#2D0845', markeredgewidth=1.5,
        zorder=11)

    # For anterior (using circles)
    for i, tp in enumerate(timepoints):
        # Filter data for this timepoint and region
        tp_data = melted_data[(melted_data['timepoint'] == tp) & 
                             (melted_data['region'] == 'Anterior')]
        if not tp_data.empty:
            # Use the same color from the viridis palette as the boxplot
            sns.stripplot(x='timepoint', y='diff_value', 
                        data=tp_data,
                        dodge=False, jitter=0.2, size=6, alpha=0.8, ax=ax,
                        marker='o', color=palette[i])
    # For posterior (using squares)
    for i, tp in enumerate(timepoints):
        # Filter data for this timepoint and region
        tp_data = melted_data[(melted_data['timepoint'] == tp) & 
                             (melted_data['region'] == 'Posterior')]
        if not tp_data.empty:
            # Use the same color from the viridis palette as the boxplot
            sns.stripplot(x='timepoint', y='diff_value', 
                        data=tp_data,
                        dodge=False, jitter=0.2, size=6, alpha=0.8, ax=ax,
                        marker='s', color=palette[i])
    
    # Create a clean legend with exactly one entry per category
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
            markersize=8, label='Anterior', alpha=0.8),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
            markersize=8, label='Posterior', alpha=0.8) 
    ]
    # ADD LME TO LEGEND
    legend_elements += [
        Line2D([0], [0], color='#440154', linewidth=2.5, marker='o', markerfacecolor='#6B5082', markeredgecolor='#2D0845', markersize=8, markeredgewidth=1.5, label='LME Estimate'),
        # ax.plot(x_positions, y_estimates, 'o', markerfacecolor='#6B5082', 
        # markersize=10, markeredgecolor='#2D0845', markeredgewidth=1.5,
        # zorder=11)
        plt.Rectangle((0,0),1,1, facecolor='#440154', alpha=0.1, label='LME 95% CI')
    ]

    # Reduce opacity of box elements
    for patch in ax.patches:
        patch.set_alpha(0.5)
    
    # Set parameter-specific labels and title
    if parameter.lower() == "fa":
        param_name = "Fractional Anisotropy"
        unit = ""
    elif parameter.lower() == "md":
        param_name = "Mean Diffusivity"
        unit = " [mm²/s]"
    
    # Set labels and title
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel(f'{param_name} Difference (Control - Craniectomy){unit}', fontsize=12)
    ax.set_title(f'{param_name} Difference by Timepoint', fontsize=14, fontweight='bold')
    
    # Add grid for y-axis only
    ax.grid(True, axis='y', linestyle='-', alpha=0.3)
    
    # Move the legend to a better position
    ax.legend(handles=legend_elements, title='Region', loc='upper right')
    
    # Show count of patients per timepoint
    for i, tp in enumerate(timepoints):
        # Get subset of df_filtered for this timepoint
        df_tp = df_filtered[df_filtered['timepoint'] == tp]

        # Count unique patients with at least one non-null anterior or posterior value
        count = df_tp[(~df_tp[anterior_column].isna()) | (~df_tp[posterior_column].isna())]['patient_id'].nunique()

        # count = melted_data[(melted_data['timepoint'] == tp) & (~melted_data['diff_value'].isna())].shape[0] // 2
        if count > 0:
            if parameter == 'fa':
                ax.text(i, ax.get_ylim()[0] * 1.35, f"n={count}",
                    ha='center', va='bottom', fontsize=10)
            else:
                ax.text(i, ax.get_ylim()[0] * 1.125, f"n={count}",
                    ha='center', va='bottom', fontsize=10)
    
    ax.xaxis.set_label_coords(0.5, -0.125)  # Move x-axis label down
    plt.tight_layout()
    
    # Save figures to specified directories
    output_dir = "DTI_Processing_Scripts/dti_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    thesis_dir = "../Thesis/phd-thesis-template-2.4/Chapter6/Figs"
    if not os.path.exists(thesis_dir):
        os.makedirs(thesis_dir)
    
    # Save figures
    plt.savefig(f'{output_dir}/{parameter}_diff_lme_boxplot_combined.png')
    plt.savefig(f'{thesis_dir}/{parameter}_diff_lme_boxplot_combined.png', dpi=600)
    plt.close()
    
    return fig, ax


def create_timepoint_boxplot_LME_dti(df, parameter, result, timepoints=['ultra-fast', 'fast', 'acute', '3-6mo', '12-24mo'], 
                                     fixed_effects_result=None, fixed_only=False):
    """
    Create a box plot with anterior and posterior data represented as different colored points,
    but with a single boxplot per timepoint combining both anterior and posterior data.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.cm as cm
    import os

    set_publication_style()  # Assuming this function exists in your environment

    fig, ax = plt.subplots(figsize=(12, 6))

    palette = sns.color_palette("viridis", len(timepoints))

    # Column Names
    anterior_column = f"{parameter}_anterior_diff"
    posterior_column = f"{parameter}_posterior_diff"

    # We need to reshape the data to have a single "value" column
    # First, filter to only the timepoints we want
    df_filtered = df[df['timepoint'].isin(timepoints)].copy()
    
    # Create a melted dataframe for plotting
    # This reshapes the data to have a single value column with a region identifier
    melted_data = pd.melt(
        df_filtered,
        id_vars=['timepoint'], 
        value_vars=[anterior_column, posterior_column],
        var_name='region_col',
        value_name='diff_value'
    )
    
    # Create a cleaner region column
    melted_data['region'] = melted_data['region_col'].apply(
        lambda x: 'Anterior' if 'anterior' in x else 'Posterior'
    )
    
    # Ensure timepoints are in the correct order
    melted_data['timepoint'] = pd.Categorical(melted_data['timepoint'],
                                             categories=timepoints,
                                             ordered=True)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Create lists to track which timepoints have small sample sizes
    small_sample_tps = []
    regular_sample_tps = []
    
    for tp in timepoints:
        # Count unique subjects in this timepoint (assuming one row per subject per timepoint)
        tp_data = melted_data[melted_data['timepoint'] == tp]
        tp_count = len(df[df['timepoint'] == tp])  # Original count from df
        if tp_count < 5:
            small_sample_tps.append(tp)
        else:
            regular_sample_tps.append(tp)
    
    # Create a copy for regular samples
    melted_regular = melted_data[melted_data['timepoint'].isin(regular_sample_tps)].copy()

    estimate_points = []

    for tp in timepoints:
        # Get the fixed effect estimate (relative to intercept)
        if tp == "acute":
            est = result.params["Intercept"]
            se = result.bse["Intercept"]
        else:
            coef_name = f"timepoint[T.{tp}]"
            est = result.params["Intercept"] + result.params.get(coef_name, 0)
            se = (result.bse["Intercept"] ** 2 + result.bse.get(coef_name, 0) ** 2) ** 0.5
        
        estimate_points.append((tp, est, se))


    ### PLOTTING
    # Plot regular boxplots for n >= 5
    if not melted_regular.empty:
        sns.boxplot(x='timepoint', y='diff_value', data=melted_regular,
                  palette=palette, width=0.6, ax=ax, saturation=0.7,
                  showfliers=False)
    
    # For small sample sizes (n < 5), plot just the median as a line
    for tp in small_sample_tps:
        tp_data = melted_data[melted_data['timepoint'] == tp]
        if not tp_data.empty and not tp_data['diff_value'].isna().all():
            tp_index = timepoints.index(tp)
            median_value = tp_data['diff_value'].median()
            
            # Plot median as a horizontal line
            ax.hlines(median_value, tp_index - 0.3, tp_index + 0.3,
                    color='black', linewidth=1.5, linestyle='-',
                    alpha=0.9, zorder=5)
            
    # PLOT LME RESULTS OVER THE TOP
    # for i, (tp, est, se) in enumerate(estimate_points):
    #     ax.errorbar(
    #         i, est, yerr=se, fmt='D', color='black', capsize=4, label='LME Estimate' if i == 0 else "",
    #         markersize=6, zorder=6  # Appear above everything
    #     )

    # # Add LME intercept line
    # intercept = result.params["Intercept"]
    # ax.axhline(y=intercept, color='black', linestyle='--', linewidth=1.2, alpha=0.7, label="LME Intercept")

    # Extract LME predictions

    
    x_positions = np.arange(len(timepoints))
    y_estimates = [est for _, est, _ in estimate_points]
    y_errors = [se for _, _, se in estimate_points]

    # Calculate confidence intervals
    ci_lower = np.array(y_estimates) - 1.96 * np.array(y_errors)
    ci_upper = np.array(y_estimates) + 1.96 * np.array(y_errors)

    # Extend to box edges (left edge of first box to right edge of last box)
    box_width = 0.6  # Should match your boxplot width
    x_extended = np.linspace(-box_width/2, len(timepoints)-1+box_width/2, 100)


    # Create step function instead of linear interpolation
    x_step = []
    y_step = []
    ci_lower_step = []
    ci_upper_step = []

    box_width = 0.6
    for i, (x, y, ci_l, ci_u) in enumerate(zip(x_positions, y_estimates, ci_lower, ci_upper)):
        if i == 0:
            # First timepoint: calculate half the distance to next timepoint
            half_dist_to_next = (x_positions[1] - x) / 2
            x_step.extend([x - half_dist_to_next, x + half_dist_to_next])
            y_step.extend([y, y])
            ci_lower_step.extend([ci_l, ci_l])
            ci_upper_step.extend([ci_u, ci_u])
        else:
            midpoint = (x_positions[i-1] + x) / 2
            
            # Extend previous timepoint to midpoint
            x_step.append(midpoint)
            y_step.append(y_estimates[i-1])
            ci_lower_step.append(ci_lower[i-1])
            ci_upper_step.append(ci_upper[i-1])
            
            # Step up/down at midpoint
            x_step.append(midpoint)
            y_step.append(y)
            ci_lower_step.append(ci_l)
            ci_upper_step.append(ci_u)
            
            # Continue to next midpoint or end
            if i == len(x_positions) - 1:
                # Last timepoint: extend by half the distance from previous
                half_dist_from_prev = (x - x_positions[i-1]) / 2
                x_step.append(x + half_dist_from_prev)
                y_step.append(y)
                ci_lower_step.append(ci_l)
                ci_upper_step.append(ci_u)


    # Plot step function
    if fixed_only is False:
        ax.fill_between(x_step, ci_lower_step, ci_upper_step, 
                        alpha=0.1, color='#440154', label='LME 95% CI', step='post')
        ax.plot(x_step, y_step, '-', color='#440154', linewidth=2.5, 
                label='LME Estimate', zorder=10, drawstyle='steps-post')
        ax.plot(x_positions, y_estimates, 'o', markerfacecolor='#6B5082', 
            markersize=10, markeredgecolor='#2D0845', markeredgewidth=1.5,
            zorder=11)

    # ADD FIXED EFFECTS PLOTTING IF PROVIDED
    if fixed_effects_result is not None:
        fe_estimate_points = []
        for tp in timepoints:
            # Get the fixed effect estimate (relative to intercept)
            if tp == "acute":
                est = fixed_effects_result.params["Intercept"]
                se = fixed_effects_result.bse["Intercept"]
            else:
                coef_name = f"timepoint[T.{tp}]"
                est = fixed_effects_result.params["Intercept"] + fixed_effects_result.params.get(coef_name, 0)
                se = (fixed_effects_result.bse["Intercept"] ** 2 + fixed_effects_result.bse.get(coef_name, 0) ** 2) ** 0.5
            
            fe_estimate_points.append((tp, est, se))

        fe_y_estimates = [est for _, est, _ in fe_estimate_points]
        fe_y_errors = [se for _, _, se in fe_estimate_points]
        fe_ci_lower = np.array(fe_y_estimates) - 1.96 * np.array(fe_y_errors)
        fe_ci_upper = np.array(fe_y_estimates) + 1.96 * np.array(fe_y_errors)

        # Create step function for fixed effects
        fe_x_step = []
        fe_y_step = []
        fe_ci_lower_step = []
        fe_ci_upper_step = []

        for i, (x, y, ci_l, ci_u) in enumerate(zip(x_positions, fe_y_estimates, fe_ci_lower, fe_ci_upper)):
            if i == 0:
                half_dist_to_next = (x_positions[1] - x) / 2
                fe_x_step.extend([x - half_dist_to_next, x + half_dist_to_next])
                fe_y_step.extend([y, y])
                fe_ci_lower_step.extend([ci_l, ci_l])
                fe_ci_upper_step.extend([ci_u, ci_u])
            else:
                midpoint = (x_positions[i-1] + x) / 2
                fe_x_step.append(midpoint)
                fe_y_step.append(fe_y_estimates[i-1])
                fe_ci_lower_step.append(fe_ci_lower[i-1])
                fe_ci_upper_step.append(fe_ci_upper[i-1])
                
                fe_x_step.append(midpoint)
                fe_y_step.append(y)
                fe_ci_lower_step.append(ci_l)
                fe_ci_upper_step.append(ci_u)
                
                if i == len(x_positions) - 1:
                    half_dist_from_prev = (x - x_positions[i-1]) / 2
                    fe_x_step.append(x + half_dist_from_prev)
                    fe_y_step.append(y)
                    fe_ci_lower_step.append(ci_l)
                    fe_ci_upper_step.append(ci_u)

        # Plot fixed effects with different color/style
        ax.fill_between(fe_x_step, fe_ci_lower_step, fe_ci_upper_step, 
                        alpha=0.1, color='#DE3163', label='Fixed Effects 95% CI', step='post')
        ax.plot(fe_x_step, fe_y_step, '--', color='#DE3163', linewidth=2.5, 
                label='Fixed Effects Estimate', zorder=12, drawstyle='steps-post')
        ax.plot(x_positions, fe_y_estimates, 's', markerfacecolor='#DE3163', 
            markersize=8, markeredgecolor='white', markeredgewidth=1.5,
            zorder=13)

    # For anterior (using circles)
    for i, tp in enumerate(timepoints):
        # Filter data for this timepoint and region
        tp_data = melted_data[(melted_data['timepoint'] == tp) & 
                             (melted_data['region'] == 'Anterior')]
        if not tp_data.empty:
            # Use the same color from the viridis palette as the boxplot
            sns.stripplot(x='timepoint', y='diff_value', 
                        data=tp_data,
                        dodge=False, jitter=0.2, size=6, alpha=0.8, ax=ax,
                        marker='o', color=palette[i])
    # For posterior (using squares)
    for i, tp in enumerate(timepoints):
        # Filter data for this timepoint and region
        tp_data = melted_data[(melted_data['timepoint'] == tp) & 
                             (melted_data['region'] == 'Posterior')]
        if not tp_data.empty:
            # Use the same color from the viridis palette as the boxplot
            sns.stripplot(x='timepoint', y='diff_value', 
                        data=tp_data,
                        dodge=False, jitter=0.2, size=6, alpha=0.8, ax=ax,
                        marker='s', color=palette[i])
    
    # Create a clean legend with exactly one entry per category
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
            markersize=8, label='Anterior', alpha=0.8),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
            markersize=8, label='Posterior', alpha=0.8) 
    ]
    if fixed_only is False:
        # ADD LME TO LEGEND
        legend_elements += [
            Line2D([0], [0], color='#440154', linewidth=2.5, marker='o', markerfacecolor='#6B5082', markeredgecolor='#2D0845', markersize=8, markeredgewidth=1.5, label='LME Estimate'),
            plt.Rectangle((0,0),1,1, facecolor='#440154', alpha=0.1, label='LME 95% CI')
        ]
    
    # ADD FIXED EFFECTS TO LEGEND IF PROVIDED
    if fixed_effects_result is not None:
        legend_elements += [
            Line2D([0], [0], color='#DE3163', linewidth=2.5, linestyle='--', marker='s', markerfacecolor='#DE3163', markeredgecolor='white', markersize=8, markeredgewidth=1.5, label='Fixed Effects Estimate'),
            plt.Rectangle((0,0),1,1, facecolor='#DE3163', alpha=0.1, label='Fixed Effects 95% CI')
        ]

    # Reduce opacity of box elements
    for patch in ax.patches:
        patch.set_alpha(0.5)
    
    # Set parameter-specific labels and title
    if parameter.lower() == "fa":
        param_name = "Fractional Anisotropy"
        unit = ""
    elif parameter.lower() == "md":
        param_name = "Mean Diffusivity"
        unit = " [mm²/s]"
    
    # Set labels and title
    ax.set_xlabel('Timepoint', fontsize=12)
    if parameter.lower() == 'md':
        ax.set_ylim(-0.3,0.15)
    ax.set_ylabel(f'{param_name} Difference (Control - Craniectomy){unit}', fontsize=12)
    ax.set_title(f'{param_name} Difference by Timepoint', fontsize=14, fontweight='bold')
    
    # Add grid for y-axis only
    ax.grid(True, axis='y', linestyle='-', alpha=0.3)
    
    # Move the legend to a better position
    ax.legend(handles=legend_elements, title='Region', loc='upper right')
    
    # Show count of patients per timepoint
    for i, tp in enumerate(timepoints):
        # Get subset of df_filtered for this timepoint
        df_tp = df_filtered[df_filtered['timepoint'] == tp]

        # Count unique patients with at least one non-null anterior or posterior value
        count = df_tp[(~df_tp[anterior_column].isna()) | (~df_tp[posterior_column].isna())]['patient_id'].nunique()

        # count = melted_data[(melted_data['timepoint'] == tp) & (~melted_data['diff_value'].isna())].shape[0] // 2
        if count > 0:
            if parameter == 'fa':
                ax.text(i, ax.get_ylim()[0] * 1.35, f"n={count}",
                    ha='center', va='bottom', fontsize=10)
            else:
                ax.text(i, ax.get_ylim()[0] * 1.125, f"n={count}",
                    ha='center', va='bottom', fontsize=10)
    
    ax.xaxis.set_label_coords(0.5, -0.125)  # Move x-axis label down
    plt.tight_layout()
    
    # Save figures to specified directories
    output_dir = "DTI_Processing_Scripts/dti_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    thesis_dir = "../Thesis/phd-thesis-template-2.4/Chapter6/Figs"
    if not os.path.exists(thesis_dir):
        os.makedirs(thesis_dir)
    
    # Save figures
    suffix=''
    if fixed_effects_result is not None:
        if fixed_only is False:
            suffix='_mixed_and_fixed'
        else:
            suffix='_fixed_only'
    plt.savefig(f'{output_dir}/{parameter}_diff_lme_boxplot_combined{suffix}.png')
    plt.savefig(f'{thesis_dir}/{parameter}_diff_lme_boxplot_combined{suffix}.png', dpi=600)
    plt.close()
    
    return fig, ax

def data_availability_matrix(data, timepoints, diff_column='fa_anterior_diff', filename='data_availability.png'):
    """
    Create a data availability matrix for the given timepoints.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data in long format with patient_id and timepoint columns
    timepoints : list
        List of timepoints to include in the matrix
    diff_column : str
        Column to check for data availability
    filename : str
        Name of the output file
        
    Returns:
    --------
    availability_matrix : pandas.DataFrame
        Matrix showing the number of patients with data for each pair of timepoints
    """
    # Create an empty matrix with timepoints as index and columns
    availability_matrix = pd.DataFrame(index=timepoints, columns=timepoints, dtype=float)
    
    # Filter data to only include rows where diff_column is not null
    valid_data = data[data[diff_column].notna()]
    
    # Fill the matrix with counts
    for time1 in timepoints:
        for time2 in timepoints:
            if time1 == time2:
                # Diagonal: number of patients with data for this timepoint
                patients_at_time = valid_data[valid_data['timepoint'] == time1]['patient_id'].nunique()
                availability_matrix.loc[time1, time2] = patients_at_time
            else:
                # Off-diagonal: number of patients with data for both timepoints
                patients_at_time1 = set(valid_data[valid_data['timepoint'] == time1]['patient_id'])
                patients_at_time2 = set(valid_data[valid_data['timepoint'] == time2]['patient_id'])
                common_patients = patients_at_time1.intersection(patients_at_time2)
                availability_matrix.loc[time1, time2] = len(common_patients)
    
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
    plt.savefig(f"DTI_Processing_Scripts/dti_plots/{filename}")
    plt.savefig(f"../Thesis/phd-thesis-template-2.4/Chapter6/Figs/{filename}", dpi=600)
    plt.close()
    
    return availability_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.patches as mpatches

def create_fa_area_correlation_plot(df, show_timepoints=True):
    """
    Create correlation plots between FA differences and herniation area.
    
    Parameters:
    df: DataFrame with columns ['fa_anterior_diff', 'fa_posterior_diff', 'area_diff', 'timepoint', 'patient_id']
    show_timepoints: Boolean, whether to color by timepoint
    """
    
    # Set publication style (assuming this function exists)
    # set_publication_style()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Color palette
    if show_timepoints:
        timepoints = df['timepoint'].unique()
        palette = sns.color_palette("viridis", len(timepoints))
        color_mapping = dict(zip(timepoints, palette))
    else:
        color = '#2E86AB'  # Single color if not showing timepoints
    
    # Plot 1: FA Anterior vs Area
    ax1 = axes[0]
    
    if show_timepoints:
        for tp in timepoints:
            tp_data = df[df['timepoint'] == tp]
            ax1.scatter(tp_data['fa_anterior_diff'], tp_data['area_diff'], 
                       color=color_mapping[tp], alpha=0.7, s=60, 
                       label=tp, edgecolors='white', linewidth=0.5)
    else:
        ax1.scatter(df['fa_anterior_diff'], df['area_diff'], 
                   color=color, alpha=0.7, s=60, 
                   edgecolors='white', linewidth=0.5)
    
    # Add regression line and statistics
    x_ant = df['fa_anterior_diff'].dropna()
    y_ant = df.loc[x_ant.index, 'area_diff'].dropna()
    if len(x_ant) > 2:
        slope_ant, intercept_ant, r_ant, p_ant, se_ant = stats.linregress(x_ant, y_ant)
        line_x = np.linspace(x_ant.min(), x_ant.max(), 100)
        line_y = slope_ant * line_x + intercept_ant
        ax1.plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2)
        
        # Add correlation info
        ax1.text(0.05, 0.95, f'r = {r_ant:.3f}\np = {p_ant:.3f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('FA Anterior Difference\n(Control - Craniectomy)', fontsize=12)
    ax1.set_ylabel('Herniation Area Difference [mm²]', fontsize=12)
    ax1.set_title('FA Anterior vs Herniation Area', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
    # Plot 2: FA Posterior vs Area
    ax2 = axes[1]
    
    if show_timepoints:
        for tp in timepoints:
            tp_data = df[df['timepoint'] == tp]
            ax2.scatter(tp_data['fa_posterior_diff'], tp_data['area_diff'], 
                       color=color_mapping[tp], alpha=0.7, s=60, 
                       label=tp, edgecolors='white', linewidth=0.5)
    else:
        ax2.scatter(df['fa_posterior_diff'], df['area_diff'], 
                   color=color, alpha=0.7, s=60, 
                   edgecolors='white', linewidth=0.5)
    
    # Add regression line and statistics
    x_post = df['fa_posterior_diff'].dropna()
    y_post = df.loc[x_post.index, 'area_diff'].dropna()
    if len(x_post) > 2:
        slope_post, intercept_post, r_post, p_post, se_post = stats.linregress(x_post, y_post)
        line_x = np.linspace(x_post.min(), x_post.max(), 100)
        line_y = slope_post * line_x + intercept_post
        ax2.plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2)
        
        # Add correlation info
        ax2.text(0.05, 0.95, f'r = {r_post:.3f}\np = {p_post:.3f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('FA Posterior Difference\n(Control - Craniectomy)', fontsize=12)
    ax2.set_ylabel('Herniation Area Difference [mm²]', fontsize=12)
    ax2.set_title('FA Posterior vs Herniation Area', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
    # Add legend if showing timepoints
    if show_timepoints:
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, title='Timepoint', 
                  bbox_to_anchor=(1.05, 0.5), loc='center left')
    
    plt.tight_layout()
    
    # Save figures
    output_dir = "DTI_Processing_Scripts/dti_plots"
    thesis_dir = "../Thesis/phd-thesis-template-2.4/Chapter6/Figs"
    
    timepoint_suffix = "_by_timepoint" if show_timepoints else ""
    plt.savefig(f'{output_dir}/fa_area_correlation{timepoint_suffix}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{thesis_dir}/fa_area_correlation{timepoint_suffix}.png', dpi=600, bbox_inches='tight')
    
    return fig, axes

# Alternative: Combined plot showing model predictions
def create_fa_area_model_visualization(df, model_results):
    """
    Create a visualization showing model predictions alongside data
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create combined FA measure (you might want to adjust this)
    df['fa_combined'] = df['fa_anterior_diff']  # Focus on anterior since it's significant
    
    # Scatter plot
    scatter = ax.scatter(df['fa_combined'], df['area_diff'], 
                        c=df['timepoint'].astype('category').cat.codes,
                        cmap='viridis', alpha=0.7, s=60, 
                        edgecolors='white', linewidth=0.5)
    
    # Add model prediction line
    x_range = np.linspace(df['fa_combined'].min(), df['fa_combined'].max(), 100)
    # Using coefficient from Model 1 (4747.88)
    y_pred = model_results['intercept'] + model_results['fa_anterior_coef'] * x_range
    ax.plot(x_range, y_pred, 'r-', linewidth=3, alpha=0.8, 
            label=f'LME Model: β = {model_results["fa_anterior_coef"]:.0f}')
    
    # Add confidence interval if available
    # y_pred_ci = ... (calculate from model if needed)
    
    ax.set_xlabel('FA Anterior Difference (Control - Craniectomy)', fontsize=12)
    ax.set_ylabel('Herniation Area Difference [mm²]', fontsize=12)
    ax.set_title('Linear Mixed Effects Model: FA-Area Correlation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax.legend()
    
    # Add colorbar for timepoints
    cbar = plt.colorbar(scatter)
    cbar.set_label('Timepoint', rotation=270, labelpad=15)
    
    plt.tight_layout()
    return fig, ax

def create_area_predicts_fa_plot(df, result4, result5, show_combined=True, timepoints=["ultra-fast", "fast", "acute", "3-6mo", "12-24mo"]):
    """
    Create scatter plot showing how herniation area predicts FA differences.
    Based on mixed effects models: area_diff predicts fa_anterior_diff and fa_posterior_diff
    
    Parameters:
    df: DataFrame with columns ['fa_anterior_diff', 'fa_posterior_diff', 'area_diff', 'timepoint', 'patient_id']
    result4: Fitted model result for fa_anterior_diff ~ area_diff  
    result5: Fitted model result for fa_posterior_diff ~ area_diff
    show_combined: Boolean, whether to show combined plot (True) or separate subplots (False)
    """
    
    # Set publication style 
    set_publication_style()

    # set up timepoint colours
    palette = sns.color_palette("viridis", len(timepoints))
    color_mapping = dict(zip(timepoints, palette))

    # Save figures
    output_dir = "DTI_Processing_Scripts/dti_plots"
    thesis_dir = "../Thesis/phd-thesis-template-2.4/Chapter6/Figs"
    
    if show_combined:
        # Single plot with both regions
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot anterior data (circles)
        # anterior_data = df.dropna(subset=['fa_anterior_diff', 'area_diff'])
        # ax.scatter(anterior_data['area_diff'], anterior_data['fa_anterior_diff'], 
        #           marker='o', s=80, alpha=0.7, color='#1f77b4', 
        #           edgecolors='white', linewidth=1, label='Anterior')
        # Plot anterior data (circles) - colored by timepoint
        anterior_data = df.dropna(subset=['fa_anterior_diff', 'area_diff'])
        for tp in timepoints:
            tp_data = anterior_data[anterior_data['timepoint'] == tp]
            if not tp_data.empty:
                ax.scatter(tp_data['area_diff'], tp_data['fa_anterior_diff'], 
                        marker='o', s=40, alpha=0.4, color=color_mapping[tp], 
                        edgecolors='white', linewidth=0.5)
        
        # Plot posterior data (squares)  
        # Plot posterior data (squares) - colored by timepoint
        posterior_data = df.dropna(subset=['fa_posterior_diff', 'area_diff'])
        for tp in timepoints:
            tp_data = posterior_data[posterior_data['timepoint'] == tp]
            if not tp_data.empty:
                ax.scatter(tp_data['area_diff'], tp_data['fa_posterior_diff'],
                        marker='s', s=40, alpha=0.4, color=color_mapping[tp],
                        edgecolors='white', linewidth=0.5)
        
        # Get area range for regression lines
        area_range = np.linspace(df['area_diff'].min(), df['area_diff'].max(), 100)
        
        # Add anterior regression line with CI
        if result4 is not None:
            # Get model coefficients
            intercept_ant = result4.params['Intercept']
            slope_ant = result4.params['area_diff']
            
            # Predict values
            predicted_ant = intercept_ant + slope_ant * area_range
            
            # Extract confidence intervals (following boxplot code pattern)
            anterior_se = result4.bse['Intercept']  # Standard error of intercept
            
            # Calculate confidence intervals for the regression line
            ci_lower_ant = predicted_ant - 1.96 * anterior_se
            ci_upper_ant = predicted_ant + 1.96 * anterior_se
            
            # Plot regression line and confidence interval
            ax.fill_between(area_range, ci_lower_ant, ci_upper_ant, 
                            color='#440154', alpha=0.1, label='Anterior 95% CI')
            ax.plot(area_range, predicted_ant, '-', color='#440154', linewidth=1.5, 
                    alpha=0.8, label='Anterior Regression', zorder=10)
        # Add posterior regression line with CI
        # Add posterior regression line with CI
        if result5 is not None:
            # Get model coefficients
            intercept_post = result5.params['Intercept']
            slope_post = result5.params['area_diff']
            
            # Predict values
            predicted_post = intercept_post + slope_post * area_range
            
            # Extract confidence intervals (following boxplot code pattern)
            posterior_se = result5.bse['Intercept']  # Standard error of intercept
            
            # Calculate confidence intervals for the regression line
            ci_lower_post = predicted_post - 1.96 * posterior_se
            ci_upper_post = predicted_post + 1.96 * posterior_se
            
            # Plot regression line and confidence interval
            ax.fill_between(area_range, ci_lower_post, ci_upper_post, 
                            color='#73D055', alpha=0.1, label='Posterior 95% CI')
            ax.plot(area_range, predicted_post, '-', color='#73D055', linewidth=1.5, 
                    alpha=0.8, label='Posterior Regression', zorder=10)
        
        # Add statistics text box
        if result4 is not None and result5 is not None:
            stats_text = f'Anterior: β = {result4.params["area_diff"]:.2e}, p = {result4.pvalues["area_diff"]:.3f}\n'
            stats_text += f'Posterior: β = {result5.params["area_diff"]:.2e}, p = {result5.pvalues["area_diff"]:.3f}'
            
            # ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, 
            #     verticalalignment='bottom', horizontalalignment='right', fontsize=10,
            #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.text(0.0125, 0.995, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='left', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_xlabel('Herniation Area [mm²]', fontsize=14)
        ax.set_ylabel('FA Difference\n(Control - Craniectomy)', fontsize=14)
        ax.set_title('Herniation Area Predicts FA Changes', fontsize=16, fontweight='bold')
        ax.grid(False) #alpha=0.3
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.1)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.1)
        ax.set_xlim(df['area_diff'].min(), df['area_diff'].max() + 50)

        # ax.legend(fontsize=12)

        # Create legend for timepoints and regression lines
        from matplotlib.lines import Line2D
        legend_elements = [
            # Timepoint legend entries
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[tp], 
                markersize=6, alpha=0.6, label=tp) for tp in timepoints
        ] + [
            # Regression line legend entries
            Line2D([0], [0], color='#440154', linewidth=1.5, label='Anterior Regression'),
            plt.Rectangle((0,0),1,1, facecolor='#440154', alpha=0.1, label='Anterior 95% CI'),
            Line2D([0], [0], color='#73D055', linewidth=1.5, label='Posterior Regression'),
            plt.Rectangle((0,0),1,1, facecolor='#73D055', alpha=0.1, label='Posterior 95% CI')
        ]

        ax.legend(handles=legend_elements, fontsize=10, loc='upper left', 
                  bbox_to_anchor=(0, 0.95))
        
    else:
        # Separate subplots - create two separate figures
        
        # Figure 1: Anterior
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        anterior_data = df.dropna(subset=['fa_anterior_diff', 'area_diff'])
        
        # Plot anterior data (circles) - colored by timepoint
        for tp in timepoints:
            tp_data = anterior_data[anterior_data['timepoint'] == tp]
            if not tp_data.empty:
                ax1.scatter(tp_data['area_diff'], tp_data['fa_anterior_diff'], 
                        marker='o', s=40, alpha=0.4, color=color_mapping[tp], 
                        edgecolors='white', linewidth=0.5)
        
        # Get area range for regression lines
        area_range = np.linspace(df['area_diff'].min(), df['area_diff'].max(), 100)
        
        if result4 is not None:
            # Get model coefficients
            intercept_ant = result4.params['Intercept']
            slope_ant = result4.params['area_diff']
            
            # Predict values
            predicted_ant = intercept_ant + slope_ant * area_range
            
            # Extract confidence intervals (following boxplot code pattern)
            anterior_se = result4.bse['Intercept']  # Standard error of intercept
            
            # Calculate confidence intervals for the regression line
            ci_lower_ant = predicted_ant - 1.96 * anterior_se
            ci_upper_ant = predicted_ant + 1.96 * anterior_se
            
            # Plot regression line and confidence interval
            ax1.fill_between(area_range, ci_lower_ant, ci_upper_ant, 
                            color='#440154', alpha=0.2, label='Anterior 95% CI')
            ax1.plot(area_range, predicted_ant, '-', color='#440154', linewidth=2, 
                    alpha=0.8, label='Anterior Regression', zorder=10)
            
            # Add statistics
            stats_text = f'β = {slope_ant:.2e}\np = {result4.pvalues["area_diff"]:.3f}'
            ax1.text(0.0125, 0.995, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', horizontalalignment='left', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    
        
        ax1.set_xlabel('Herniation Area [mm²]', fontsize=12)
        ax1.set_ylabel('Anterior FA Difference\n(Control - Craniectomy)', fontsize=12)
        ax1.set_title('Anterior FA vs Herniation Area', fontsize=14, fontweight='bold')
        ax1.grid(False)
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.1)
        ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.1)
        ax1.set_xlim(df['area_diff'].min(), df['area_diff'].max() + 50)
        
        # Create legend for timepoints and regression line
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[tp], 
                markersize=6, alpha=0.6, label=tp) for tp in timepoints
        ] + [
            Line2D([0], [0], color='#440154', linewidth=2, label='Anterior Regression'),
            plt.Rectangle((0,0),1,1, facecolor='#440154', alpha=0.2, label='Anterior 95% CI')
        ]
        ax1.legend(handles=legend_elements, fontsize=10, loc='upper left', 
                  bbox_to_anchor=(0, 0.95))
        
        plt.tight_layout()
        
        # Save anterior figure
        plt.savefig(f'{output_dir}/area_predicts_fa_anterior.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{thesis_dir}/area_predicts_fa_anterior.png', dpi=600, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Posterior
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        posterior_data = df.dropna(subset=['fa_posterior_diff', 'area_diff'])
        
        # Plot posterior data (squares) - colored by timepoint
        for tp in timepoints:
            tp_data = posterior_data[posterior_data['timepoint'] == tp]
            if not tp_data.empty:
                ax2.scatter(tp_data['area_diff'], tp_data['fa_posterior_diff'],
                        marker='s', s=40, alpha=0.4, color=color_mapping[tp],
                        edgecolors='white', linewidth=0.5)
        
        if result5 is not None:
            # Get model coefficients
            intercept_post = result5.params['Intercept']
            slope_post = result5.params['area_diff']
            
            # Predict values
            predicted_post = intercept_post + slope_post * area_range
            
            # Extract confidence intervals (following boxplot code pattern)
            posterior_se = result5.bse['Intercept']  # Standard error of intercept
            
            # Calculate confidence intervals for the regression line
            ci_lower_post = predicted_post - 1.96 * posterior_se
            ci_upper_post = predicted_post + 1.96 * posterior_se
            
            # Plot regression line and confidence interval
            ax2.fill_between(area_range, ci_lower_post, ci_upper_post, 
                            color='#73D055', alpha=0.2, label='Posterior 95% CI')
            ax2.plot(area_range, predicted_post, '-', color='#73D055', linewidth=2, 
                    alpha=0.8, label='Posterior Regression', zorder=10)
            
            # Add statistics
            stats_text = f'β = {slope_post:.2e}\np = {result5.pvalues["area_diff"]:.3f}'
            ax2.text(0.0125, 0.995, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', horizontalalignment='left', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Herniation Area [mm²]', fontsize=12)
        ax2.set_ylabel('Posterior FA Difference\n(Control - Craniectomy)', fontsize=12)
        ax2.set_title('Posterior FA vs Herniation Area', fontsize=14, fontweight='bold')
        ax2.grid(False)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.1)
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.1)
        ax2.set_xlim(df['area_diff'].min(), df['area_diff'].max() + 50)
        
        # Create legend for timepoints and regression line
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor=color_mapping[tp], 
                markersize=6, alpha=0.6, label=tp) for tp in timepoints
        ] + [
            Line2D([0], [0], color='#73D055', linewidth=2, label='Posterior Regression'),
            plt.Rectangle((0,0),1,1, facecolor='#73D055', alpha=0.2, label='Posterior 95% CI')
        ]
        ax2.legend(handles=legend_elements, fontsize=10, loc='upper left', 
                  bbox_to_anchor=(0, 0.95))
        
        plt.tight_layout()
        
        # Save posterior figure
        plt.savefig(f'{output_dir}/area_predicts_fa_posterior.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{thesis_dir}/area_predicts_fa_posterior.png', dpi=600, bbox_inches='tight')
        plt.close()
        
        return  # Exit early since we created separate figures

    
    plt.tight_layout()
    
    # Save figures
    # output_dir = "DTI_Processing_Scripts/dti_plots"
    # thesis_dir = "../Thesis/phd-thesis-template-2.4/Chapter6/Figs"
    
    plot_type = "combined" if show_combined else "separate"
    plt.savefig(f'{output_dir}/area_predicts_fa_{plot_type}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{thesis_dir}/area_predicts_fa_{plot_type}.png', dpi=600, bbox_inches='tight')
    
    plt.close()

    return


def create_area_predicts_md_plot(df, result4, result5, show_combined=True, timepoints=["ultra-fast", "fast", "acute", "3-6mo", "12-24mo"]):
    """
    Create scatter plot showing how herniation area predicts MD differences.
    Based on mixed effects models: area_diff predicts md_anterior_diff and md_posterior_diff
    
    Parameters:
    df: DataFrame with columns ['md_anterior_diff', 'md_posterior_diff', 'area_diff', 'timepoint', 'patient_id']
    result4: Fitted model result for md_anterior_diff ~ area_diff  
    result5: Fitted model result for md_posterior_diff ~ area_diff
    show_combined: Boolean, whether to show combined plot (True) or separate subplots (False)
    """
    
    # Set publication style 
    set_publication_style()

    # set up timepoint colours
    palette = sns.color_palette("viridis", len(timepoints))
    color_mapping = dict(zip(timepoints, palette))

    # Define output directories
    output_dir = "DTI_Processing_Scripts/dti_plots"
    thesis_dir = "../Thesis/phd-thesis-template-2.4/Chapter6/Figs"
    
    if show_combined:
        # Single plot with both regions
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot anterior data (circles) - colored by timepoint
        anterior_data = df.dropna(subset=['md_anterior_diff', 'area_diff'])
        for tp in timepoints:
            tp_data = anterior_data[anterior_data['timepoint'] == tp]
            if not tp_data.empty:
                ax.scatter(tp_data['area_diff'], tp_data['md_anterior_diff'], 
                        marker='o', s=40, alpha=0.4, color=color_mapping[tp], 
                        edgecolors='white', linewidth=0.5)
        
        # Plot posterior data (squares) - colored by timepoint
        posterior_data = df.dropna(subset=['md_posterior_diff', 'area_diff'])
        for tp in timepoints:
            tp_data = posterior_data[posterior_data['timepoint'] == tp]
            if not tp_data.empty:
                ax.scatter(tp_data['area_diff'], tp_data['md_posterior_diff'],
                        marker='s', s=40, alpha=0.4, color=color_mapping[tp],
                        edgecolors='white', linewidth=0.5)
        
        # Get area range for regression lines
        area_range = np.linspace(df['area_diff'].min(), df['area_diff'].max(), 100)
        
        # Add anterior regression line with CI
        if result4 is not None:
            # Get model coefficients
            intercept_ant = result4.params['Intercept']
            slope_ant = result4.params['area_diff']
            
            # Predict values
            predicted_ant = intercept_ant + slope_ant * area_range
            
            # Extract confidence intervals (following boxplot code pattern)
            anterior_se = result4.bse['Intercept']  # Standard error of intercept
            
            # Calculate confidence intervals for the regression line
            ci_lower_ant = predicted_ant - 1.96 * anterior_se
            ci_upper_ant = predicted_ant + 1.96 * anterior_se
            
            # Plot regression line and confidence interval
            ax.fill_between(area_range, ci_lower_ant, ci_upper_ant, 
                            color='#31688E', alpha=0.1, label='Anterior 95% CI')
            ax.plot(area_range, predicted_ant, '-', color='#31688E', linewidth=1.5, 
                    alpha=0.8, label='Anterior Regression', zorder=10)
        
        # Add posterior regression line with CI
        if result5 is not None:
            # Get model coefficients
            intercept_post = result5.params['Intercept']
            slope_post = result5.params['area_diff']
            
            # Predict values
            predicted_post = intercept_post + slope_post * area_range
            
            # Extract confidence intervals (following boxplot code pattern)
            posterior_se = result5.bse['Intercept']  # Standard error of intercept
            
            # Calculate confidence intervals for the regression line
            ci_lower_post = predicted_post - 1.96 * posterior_se
            ci_upper_post = predicted_post + 1.96 * posterior_se
            
            # Plot regression line and confidence interval
            ax.fill_between(area_range, ci_lower_post, ci_upper_post, 
                            color='#FDE725', alpha=0.1, label='Posterior 95% CI')
            ax.plot(area_range, predicted_post, '-', color='#FDE725', linewidth=1.5, 
                    alpha=0.8, label='Posterior Regression', zorder=10)
        
        # Add statistics text box
        if result4 is not None and result5 is not None:
            stats_text = f'Anterior: β = {result4.params["area_diff"]:.2e}, p = {result4.pvalues["area_diff"]:.3f}\n'
            stats_text += f'Posterior: β = {result5.params["area_diff"]:.2e}, p = {result5.pvalues["area_diff"]:.3f}'
            
            ax.text(0.0125, 0.995, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='left', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_xlabel('Herniation Area [mm²]', fontsize=14)
        ax.set_ylabel('MD Difference\n(Control - Craniectomy) [mm²/s]', fontsize=14)
        ax.set_title('Herniation Area Predicts MD Changes', fontsize=16, fontweight='bold')
        ax.grid(False)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.1)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.1)
        ax.set_xlim(df['area_diff'].min(), df['area_diff'].max() + 50)
        ax.set_ylim(-0.3, 0.25)

        # Create legend for timepoints and regression lines
        from matplotlib.lines import Line2D
        legend_elements = [
            # Timepoint legend entries
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[tp], 
                markersize=6, alpha=0.6, label=tp) for tp in timepoints
        ] + [
            # Regression line legend entries
            Line2D([0], [0], color='#31688E', linewidth=1.5, label='Anterior Regression'),
            plt.Rectangle((0,0),1,1, facecolor='#31688E', alpha=0.1, label='Anterior 95% CI'),
            Line2D([0], [0], color='#FDE725', linewidth=1.5, label='Posterior Regression'),
            plt.Rectangle((0,0),1,1, facecolor='#FDE725', alpha=0.1, label='Posterior 95% CI')
        ]

        ax.legend(handles=legend_elements, fontsize=10, loc='upper left', 
                  bbox_to_anchor=(0, 0.95))
        
    else:
        # Separate subplots - create two separate figures
        
        # Figure 1: Anterior
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        anterior_data = df.dropna(subset=['md_anterior_diff', 'area_diff'])
        
        # Plot anterior data (circles) - colored by timepoint
        for tp in timepoints:
            tp_data = anterior_data[anterior_data['timepoint'] == tp]
            if not tp_data.empty:
                ax1.scatter(tp_data['area_diff'], tp_data['md_anterior_diff'], 
                        marker='o', s=40, alpha=0.4, color=color_mapping[tp], 
                        edgecolors='white', linewidth=0.5)
        
        # Get area range for regression lines
        area_range = np.linspace(df['area_diff'].min(), df['area_diff'].max(), 100)
        
        if result4 is not None:
            # Get model coefficients
            intercept_ant = result4.params['Intercept']
            slope_ant = result4.params['area_diff']
            
            # Predict values
            predicted_ant = intercept_ant + slope_ant * area_range
            
            # Extract confidence intervals (following boxplot code pattern)
            anterior_se = result4.bse['Intercept']  # Standard error of intercept
            
            # Calculate confidence intervals for the regression line
            ci_lower_ant = predicted_ant - 1.96 * anterior_se
            ci_upper_ant = predicted_ant + 1.96 * anterior_se
            
            # Plot regression line and confidence interval
            ax1.fill_between(area_range, ci_lower_ant, ci_upper_ant, 
                            color='#31688E', alpha=0.2, label='Anterior 95% CI')
            ax1.plot(area_range, predicted_ant, '-', color='#31688E', linewidth=2, 
                    alpha=0.8, label='Anterior Regression', zorder=10)
            
            # Add statistics
            stats_text = f'β = {slope_ant:.2e}\np = {result4.pvalues["area_diff"]:.3f}'
            ax1.text(0.0125, 0.995, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', horizontalalignment='left', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_xlabel('Herniation Area [mm²]', fontsize=12)
        ax1.set_ylabel('Anterior MD Difference\n(Control - Craniectomy) [mm²/s]', fontsize=12)
        ax1.set_title('Anterior MD vs Herniation Area', fontsize=14, fontweight='bold')
        ax1.grid(False)
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.1)
        ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.1)
        ax1.set_xlim(df['area_diff'].min(), df['area_diff'].max() + 50)
        ax1.set_ylim(-0.3, 0.25)
        
        # Create legend for timepoints and regression line
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[tp], 
                markersize=6, alpha=0.6, label=tp) for tp in timepoints
        ] + [
            Line2D([0], [0], color='#31688E', linewidth=2, label='Anterior Regression'),
            plt.Rectangle((0,0),1,1, facecolor='#31688E', alpha=0.2, label='Anterior 95% CI')
        ]
        ax1.legend(handles=legend_elements, fontsize=10, loc='upper left', 
                  bbox_to_anchor=(0, 0.95))
        
        plt.tight_layout()
        
        # Save anterior figure
        plt.savefig(f'{output_dir}/area_predicts_md_anterior.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{thesis_dir}/area_predicts_md_anterior.png', dpi=600, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Posterior
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        posterior_data = df.dropna(subset=['md_posterior_diff', 'area_diff'])
        
        # Plot posterior data (squares) - colored by timepoint
        for tp in timepoints:
            tp_data = posterior_data[posterior_data['timepoint'] == tp]
            if not tp_data.empty:
                ax2.scatter(tp_data['area_diff'], tp_data['md_posterior_diff'],
                        marker='s', s=40, alpha=0.4, color=color_mapping[tp],
                        edgecolors='white', linewidth=0.5)
        
        if result5 is not None:
            # Get model coefficients
            intercept_post = result5.params['Intercept']
            slope_post = result5.params['area_diff']
            
            # Predict values
            predicted_post = intercept_post + slope_post * area_range
            
            # Extract confidence intervals (following boxplot code pattern)
            posterior_se = result5.bse['Intercept']  # Standard error of intercept
            
            # Calculate confidence intervals for the regression line
            ci_lower_post = predicted_post - 1.96 * posterior_se
            ci_upper_post = predicted_post + 1.96 * posterior_se
            
            # Plot regression line and confidence interval
            ax2.fill_between(area_range, ci_lower_post, ci_upper_post, 
                            color='#FDE725', alpha=0.2, label='Posterior 95% CI')
            ax2.plot(area_range, predicted_post, '-', color='#FDE725', linewidth=2, 
                    alpha=0.8, label='Posterior Regression', zorder=10)
            
            # Add statistics
            stats_text = f'β = {slope_post:.2e}\np = {result5.pvalues["area_diff"]:.3f}'
            ax2.text(0.0125, 0.995, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', horizontalalignment='left', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Herniation Area [mm²]', fontsize=12)
        ax2.set_ylabel('Posterior MD Difference\n(Control - Craniectomy) [mm²/s]', fontsize=12)
        ax2.set_title('Posterior MD vs Herniation Area', fontsize=14, fontweight='bold')
        ax2.grid(False)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.1)
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.1)
        ax2.set_xlim(df['area_diff'].min(), df['area_diff'].max() + 50)
        ax2.set_ylim(-0.3, 0.25)
        
        # Create legend for timepoints and regression line
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor=color_mapping[tp], 
                markersize=6, alpha=0.6, label=tp) for tp in timepoints
        ] + [
            Line2D([0], [0], color='#FDE725', linewidth=2, label='Posterior Regression'),
            plt.Rectangle((0,0),1,1, facecolor='#FDE725', alpha=0.2, label='Posterior 95% CI')
        ]
        ax2.legend(handles=legend_elements, fontsize=10, loc='upper left', 
                  bbox_to_anchor=(0, 0.95))
        
        plt.tight_layout()
        
        # Save posterior figure
        plt.savefig(f'{output_dir}/area_predicts_md_posterior.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{thesis_dir}/area_predicts_md_posterior.png', dpi=600, bbox_inches='tight')
        plt.close()
        
        return  # Exit early since we created separate figures

    plt.tight_layout()
    
    # Save figures
    plot_type = "combined" if show_combined else "separate"
    plt.savefig(f'{output_dir}/area_predicts_md_{plot_type}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{thesis_dir}/area_predicts_md_{plot_type}.png', dpi=600, bbox_inches='tight')
    
    plt.close()

    return


def parameter_differences(df):
    """
    Calculate differences between baseline and actual values for FA and MD parameters
    in anterior and posterior regions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing DTI data with columns for FA and MD values
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added difference columns (fa_anterior_diff, fa_posterior_diff,
        md_anterior_diff, md_posterior_diff)
    """
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Calculate FA differences (craniectomy - control)
    result_df['fa_anterior_diff'] = result_df['fa_anterior_ring_5_6_7_avg']-result_df['fa_baseline_anterior_ring_5_6_7_avg']
    result_df['fa_posterior_diff'] = result_df['fa_posterior_ring_5_6_7_avg'] - result_df['fa_baseline_posterior_ring_5_6_7_avg']
    
    # Calculate MD differences (craniectomy - control)
    result_df['md_anterior_diff'] = result_df['md_anterior_ring_5_6_7_avg'] - result_df['md_baseline_anterior_ring_5_6_7_avg']
    result_df['md_posterior_diff'] = result_df['md_posterior_ring_5_6_7_avg'] - result_df['md_baseline_posterior_ring_5_6_7_avg'] 
    
    # Print summary of the calculated differences
    print("Difference calculations:")
    for col in ['fa_anterior_diff', 'fa_posterior_diff', 'md_anterior_diff', 'md_posterior_diff']:
        valid_count = result_df[col].count()
        mean_val = result_df[col].mean()
        print(f"  {col}: {valid_count} valid values, mean = {mean_val:.6f}")
    
    return result_df

def create_timepoint_boxplot_recategorised_dti_old(df, parameter, timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']):
    """
    Create a box plot of area_diff for each timepoint with overlaid scatter points.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.cm as cm

    set_publication_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    palette = sns.color_palette("viridis", len(timepoints))

    # Column Names
    anterior_column=f"{parameter}_anterior_diff"
    posterior_column=f"{parameter}_posterior_diff"

    # Filter the dataframe to include only timepoints in the specified list
    df_filtered = df[df['timepoint'].isin(timepoints)].copy()
    
    # Ensure timepoints are in the correct order
    df_filtered['timepoint'] = pd.Categorical(df_filtered['timepoint'],
                                             categories=timepoints,
                                             ordered=True)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)


    # Plot for both anterior and posterior regions
    for region, marker, offset in [('anterior', 'o', -0.15), ('posterior', 's', 0.15)]:
        column_name = f"{parameter}_{region}_diff"
        
        # Create lists to track which timepoints have small sample sizes
        small_sample_tps = []
        regular_sample_tps = []
        
        for tp in timepoints:
            tp_data = df[df['timepoint'] == tp]
            if len(tp_data) < 5:
                small_sample_tps.append(tp)
            else:
                regular_sample_tps.append(tp)
        
        # Create a copy for regular samples
        df_regular = df_filtered[df_filtered['timepoint'].isin(regular_sample_tps)].copy()
        
        # Plot regular boxplots for n >= 5 WITHOUT positions
        if not df_regular.empty:
            sns.boxplot(x='timepoint', y=column_name, data=df_regular,
                      palette=palette, width=0.4, ax=ax, saturation=0.7,
                      showfliers=False)
        
        # For small sample sizes (n < 5), plot just the median as a line
        for tp in small_sample_tps:
            tp_data = df_filtered[df_filtered['timepoint'] == tp]
            if not tp_data.empty and not tp_data[column_name].isna().all():
                tp_index = timepoints.index(tp)
                median_value = tp_data[column_name].median()
                
                # Plot median as a horizontal line
                ax.hlines(median_value, tp_index - 0.2, tp_index + 0.2,
                         color='black', linewidth=1.0, linestyle='-',
                         alpha=0.9, zorder=5)
                
         # Add scatter points for all timepoints
        sns.stripplot(x='timepoint', y=column_name, data=df_filtered,
                    marker=marker, palette=palette, jitter=True, 
                    size=6, alpha=0.8, ax=ax,
                    label=f"{region.capitalize()}")
                
        
    
    # Reduce opacity of box elements after creation
    for patch in ax.patches:
        patch.set_alpha(0.5)
    
    # Set parameter-specific labels and title
    if parameter.lower() == "fa":
        param_name = "Fractional Anisotropy"
        unit = ""
    elif parameter.lower() == "md":
        param_name = "Mean Diffusivity"
        unit = " [mm²/s]" 


    # Set labels and title
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel(f'{param_name} Difference (Control - Craniectomy){unit}', fontsize=12)
    ax.set_title(f'{param_name} Difference by Timepoint', fontsize=14, fontweight='bold')
    
    # Add grid for y-axis only
    ax.grid(True, axis='y', linestyle='-', alpha=0.3)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicates from legend
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Region')
    
    # Show count of patients per timepoint
    for i, tp in enumerate(timepoints):
        count = len(df[df['timepoint'] == tp])
        if count > 0:
            ax.text(i, ax.get_ylim()[0] * 1.05, f"n={count}",
                   ha='center', va='bottom', fontsize=10)
    
    ax.xaxis.set_label_coords(0.5, -0.125) # Move x-axis label down
    plt.tight_layout()

    # Save figures to specified directories
    output_dir = "DTI_Processing_Scripts/dti_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    thesis_dir = "../Thesis/phd-thesis-template-2.4/Chapter6/Figs"
    if not os.path.exists(thesis_dir):
        os.makedirs(thesis_dir)
    
    # Save figures
    plt.savefig(f'{output_dir}/{parameter}_diff_boxplot.png')
    plt.savefig(f'{thesis_dir}/{parameter}_diff_boxplot.png', dpi=600)
    plt.close()
    
    return fig, ax

def create_timepoint_boxplot_recategorised_dti(df, parameter, timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']):
    """
    Create a box plot with anterior and posterior data represented as different colored points,
    but with a single boxplot per timepoint combining both anterior and posterior data.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.cm as cm
    import os

    set_publication_style()  # Assuming this function exists in your environment

    fig, ax = plt.subplots(figsize=(12, 6))

    palette = sns.color_palette("viridis", len(timepoints))

    # Column Names
    anterior_column = f"{parameter}_anterior_diff"
    posterior_column = f"{parameter}_posterior_diff"

    # We need to reshape the data to have a single "value" column
    # First, filter to only the timepoints we want
    df_filtered = df[df['timepoint'].isin(timepoints)].copy()
    
    # Create a melted dataframe for plotting
    # This reshapes the data to have a single value column with a region identifier
    melted_data = pd.melt(
        df_filtered,
        id_vars=['timepoint'], 
        value_vars=[anterior_column, posterior_column],
        var_name='region_col',
        value_name='diff_value'
    )
    
    # Create a cleaner region column
    melted_data['region'] = melted_data['region_col'].apply(
        lambda x: 'Anterior' if 'anterior' in x else 'Posterior'
    )
    
    # Ensure timepoints are in the correct order
    melted_data['timepoint'] = pd.Categorical(melted_data['timepoint'],
                                             categories=timepoints,
                                             ordered=True)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Create lists to track which timepoints have small sample sizes
    small_sample_tps = []
    regular_sample_tps = []
    
    for tp in timepoints:
        # Count unique subjects in this timepoint (assuming one row per subject per timepoint)
        tp_data = melted_data[melted_data['timepoint'] == tp]
        tp_count = len(df[df['timepoint'] == tp])  # Original count from df
        if tp_count < 5:
            small_sample_tps.append(tp)
        else:
            regular_sample_tps.append(tp)
    
    # Create a copy for regular samples
    melted_regular = melted_data[melted_data['timepoint'].isin(regular_sample_tps)].copy()
    
    # Plot regular boxplots for n >= 5
    if not melted_regular.empty:
        sns.boxplot(x='timepoint', y='diff_value', data=melted_regular,
                  palette=palette, width=0.6, ax=ax, saturation=0.7,
                  showfliers=False)
    
    # For small sample sizes (n < 5), plot just the median as a line
    for tp in small_sample_tps:
        tp_data = melted_data[melted_data['timepoint'] == tp]
        if not tp_data.empty and not tp_data['diff_value'].isna().all():
            tp_index = timepoints.index(tp)
            median_value = tp_data['diff_value'].median()
            
            # Plot median as a horizontal line
            ax.hlines(median_value, tp_index - 0.3, tp_index + 0.3,
                    color='black', linewidth=1.5, linestyle='-',
                    alpha=0.9, zorder=5)
    
    # Add scatter points, colored by region
    # # For anterior (using circles)
    # sns.stripplot(x='timepoint', y='diff_value', 
    #             data=melted_data[melted_data['region'] == 'Anterior'],
    #             dodge=False, jitter=0.2, size=6, alpha=0.8, ax=ax,
    #             marker='o') #color='#3498db'

    # # For posterior (using squares)
    # sns.stripplot(x='timepoint', y='diff_value', 
    #             data=melted_data[melted_data['region'] == 'Posterior'],
    #             dodge=False, jitter=0.2, size=6, alpha=0.8, ax=ax,
    #             marker='s') # color='#e74c3c', 

    # For anterior (using circles)
    for i, tp in enumerate(timepoints):
        # Filter data for this timepoint and region
        tp_data = melted_data[(melted_data['timepoint'] == tp) & 
                             (melted_data['region'] == 'Anterior')]
        if not tp_data.empty:
            # Use the same color from the viridis palette as the boxplot
            sns.stripplot(x='timepoint', y='diff_value', 
                        data=tp_data,
                        dodge=False, jitter=0.2, size=6, alpha=0.8, ax=ax,
                        marker='o', color=palette[i])
    # For posterior (using squares)
    for i, tp in enumerate(timepoints):
        # Filter data for this timepoint and region
        tp_data = melted_data[(melted_data['timepoint'] == tp) & 
                             (melted_data['region'] == 'Posterior')]
        if not tp_data.empty:
            # Use the same color from the viridis palette as the boxplot
            sns.stripplot(x='timepoint', y='diff_value', 
                        data=tp_data,
                        dodge=False, jitter=0.2, size=6, alpha=0.8, ax=ax,
                        marker='s', color=palette[i])
    
    # Create a clean legend with exactly one entry per category
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
            markersize=8, label='Anterior', alpha=0.8),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
            markersize=8, label='Posterior', alpha=0.8)
    ]


    # Reduce opacity of box elements
    for patch in ax.patches:
        patch.set_alpha(0.5)
    
    # Set parameter-specific labels and title
    if parameter.lower() == "fa":
        param_name = "Fractional Anisotropy"
        unit = ""
    elif parameter.lower() == "md":
        param_name = "Mean Diffusivity"
        unit = " [mm²/s]"
    
    # Set labels and title
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel(f'{param_name} Difference (Control - Craniectomy){unit}', fontsize=12)
    ax.set_title(f'{param_name} Difference by Timepoint', fontsize=14, fontweight='bold')
    
    # Add grid for y-axis only
    ax.grid(True, axis='y', linestyle='-', alpha=0.3)
    
    # Move the legend to a better position
    ax.legend(handles=legend_elements, title='Region', loc='upper right')
    
    # Show count of patients per timepoint
    for i, tp in enumerate(timepoints):
        # Get subset of df_filtered for this timepoint
        df_tp = df_filtered[df_filtered['timepoint'] == tp]

        # Count unique patients with at least one non-null anterior or posterior value
        count = df_tp[(~df_tp[anterior_column].isna()) | (~df_tp[posterior_column].isna())]['patient_id'].nunique()

        # count = melted_data[(melted_data['timepoint'] == tp) & (~melted_data['diff_value'].isna())].shape[0] // 2
        if count > 0:
            if parameter == 'fa':
                ax.text(i, ax.get_ylim()[0] * 1.35, f"n={count}",
                    ha='center', va='bottom', fontsize=10)
            else:
                ax.text(i, ax.get_ylim()[0] * 1.125, f"n={count}",
                    ha='center', va='bottom', fontsize=10)
    
    ax.xaxis.set_label_coords(0.5, -0.125)  # Move x-axis label down
    plt.tight_layout()
    
    # Save figures to specified directories
    output_dir = "DTI_Processing_Scripts/dti_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    thesis_dir = "../Thesis/phd-thesis-template-2.4/Chapter6/Figs"
    if not os.path.exists(thesis_dir):
        os.makedirs(thesis_dir)
    
    # Save figures
    plt.savefig(f'{output_dir}/{parameter}_diff_boxplot_combined.png')
    plt.savefig(f'{thesis_dir}/{parameter}_diff_boxplot_combined.png', dpi=600)
    plt.close()
    
    return fig, ax

def create_timepoint_boxplot_recategorised_dti_single_region(df, parameter, region='anterior', timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']):
    """
    Create a box plot for a single region (anterior or posterior) by timepoint.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing the data
    parameter : str
        The parameter to plot (e.g., 'fa', 'md')
    region : str, optional
        The region to plot ('anterior' or 'posterior'), by default 'anterior'
    timepoints : list, optional
        List of timepoints to include in the plot
        
    Returns:
    --------
    fig, ax : tuple
        The figure and axis objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.cm as cm
    import os

    set_publication_style()  # Assuming this function exists in your environment

    fig, ax = plt.subplots(figsize=(12, 6))

    palette = sns.color_palette("viridis", len(timepoints))

    # Column name for the selected region
    column_name = f"{parameter}_{region}_diff"

    # Filter the dataframe to include only timepoints in the specified list
    df_filtered = df[df['timepoint'].isin(timepoints)].copy()
    
    # Ensure timepoints are in the correct order
    df_filtered['timepoint'] = pd.Categorical(df_filtered['timepoint'],
                                             categories=timepoints,
                                             ordered=True)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Create lists to track which timepoints have small sample sizes
    small_sample_tps = []
    regular_sample_tps = []
    
    for tp in timepoints:
        tp_data = df[df['timepoint'] == tp]
        if len(tp_data) < 5:
            small_sample_tps.append(tp)
        else:
            regular_sample_tps.append(tp)
    
    # Create a copy for regular samples
    df_regular = df_filtered[df_filtered['timepoint'].isin(regular_sample_tps)].copy()
    
    # Plot regular boxplots for n >= 5
    if not df_regular.empty:
        sns.boxplot(x='timepoint', y=column_name, data=df_regular,
                  palette=palette, width=0.6, ax=ax, saturation=0.7,
                  showfliers=False)
    
    # For small sample sizes (n < 5), plot just the median as a line
    for tp in small_sample_tps:
        tp_data = df_filtered[df_filtered['timepoint'] == tp]
        if not tp_data.empty and not tp_data[column_name].isna().all():
            tp_index = timepoints.index(tp)
            median_value = tp_data[column_name].median()
            
            # Plot median as a horizontal line
            ax.hlines(median_value, tp_index - 0.3, tp_index + 0.3,
                    color='black', linewidth=1.5, linestyle='-',
                    alpha=0.9, zorder=5)
    
    # Add scatter points
    sns.stripplot(x='timepoint', y=column_name, data=df_filtered,
                marker='o', palette=palette, jitter=True, 
                size=7, alpha=0.8, ax=ax)
    
    # Reduce opacity of box elements
    for patch in ax.patches:
        patch.set_alpha(0.5)
    
    # Set parameter-specific labels and title
    if parameter.lower() == "fa":
        param_name = "Fractional Anisotropy"
        unit = ""
    elif parameter.lower() == "md":
        param_name = "Mean Diffusivity"
        unit = " [mm²/s]"
    
    # Set labels and title
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel(f'{param_name} Difference (Control - Craniectomy){unit}', fontsize=12)
    ax.set_title(f'{param_name} Difference by Timepoint - {region.capitalize()} Region', 
                fontsize=14, fontweight='bold')
    
    # Add grid for y-axis only
    ax.grid(True, axis='y', linestyle='-', alpha=0.3)
    
    # Show count of patients per timepoint
    for i, tp in enumerate(timepoints):
        count = len(df[df['timepoint'] == tp])
        if count > 0:
            if parameter == 'fa':
                ax.text(i, ax.get_ylim()[0] * 1.35, f"n={count}",
                    ha='center', va='bottom', fontsize=10)
            else:
                ax.text(i, ax.get_ylim()[0] * 1.125, f"n={count}",
                    ha='center', va='bottom', fontsize=10)
    
    ax.xaxis.set_label_coords(0.5, -0.125)  # Move x-axis label down
    
    plt.tight_layout()
    
    # Save figures to specified directories
    output_dir = "DTI_Processing_Scripts/dti_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    thesis_dir = "../Thesis/phd-thesis-template-2.4/Chapter6/Figs"
    if not os.path.exists(thesis_dir):
        os.makedirs(thesis_dir)
    
    # Save figures
    plt.savefig(f'{output_dir}/{parameter}_{region}_diff_boxplot.png')
    plt.savefig(f'{thesis_dir}/{parameter}_{region}_diff_boxplot.png', dpi=600)
    plt.close()
    
    return fig, ax


def jt_test(df, parameter='fa', regions=(2,10), save_path=None, alternative='increasing', combine_regions=False):
    """
    Perform Jonckheere-Terpstra test on differences between baseline and current values
    across specified rings, and visualize the results.
    
    Args:
        df: DataFrame with the data
        parameter: The metric to analyze (e.g., 'fa', 'md')
        regions: Tuple specifying (start_ring, end_ring) to analyze
        save_path: Path to save the figure (optional)
        alternative: Direction of trend to test ('increasing', 'decreasing', or 'two-sided')
        combine_regions: Whether to combine anterior and posterior regions (True) or plot separately (False)
        
    Returns:
        Dictionary with test results and figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import re
    import scipy.stats as stats
    import seaborn as sns
    from collections import defaultdict
    
    # Import rpy2 for R integration
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    
    # Activate pandas to R conversion
    pandas2ri.activate()
    
    # Import necessary R packages
    base = importr('base')
    utils = importr('utils')
    
    # Check if PMCMRplus is installed, if not, install it
    try:
        pmcmr = importr('PMCMRplus')
    except:
        utils.chooseCRANmirror(ind=1)
        utils.install_packages('PMCMRplus')
        pmcmr = importr('PMCMRplus')
        
    # Set publication style
    try:
        # plt.style.use('seaborn-whitegrid')
        set_publication_style()
        plt.rcParams.update({
            'font.size': 14
        })
    except:
        try:
            plt.style.use('seaborn-v0_8-whitegrid')  # For newer matplotlib versions
        except:
            plt.style.use('default')  # Fallback if neither works
            
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (12, 8)
    })
    
    # Validate inputs
    parameter = parameter.lower()
    start_ring, end_ring = regions
    if not (1 <= start_ring <= 10 and 1 <= end_ring <= 10 and start_ring <= end_ring):
        raise ValueError("Ring range must be between 1-10 with start <= end")
    
    # Define the regions to analyze (always both anterior and posterior)
    regions_to_analyze = ['anterior', 'posterior']
    
    # Dictionary to store results
    results = {}
    
    # Create a figure for results
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors for regions
    colors = {'anterior': 'blue', 'posterior': 'red', 'combined': 'purple'}
    
    # Ring numbers to analyze
    ring_numbers = list(range(start_ring, end_ring + 1))
    
    if combine_regions:
        # Dictionary to store combined data for all rings
        combined_ring_data = defaultdict(list)
        
        # First, collect all data from both regions
        for curr_region in regions_to_analyze:
            for ring_num in ring_numbers:
                current_col = f"{parameter}_{curr_region}_ring_{ring_num}"
                baseline_col = f"{parameter}_baseline_{curr_region}_ring_{ring_num}"
                
                if current_col in df.columns and baseline_col in df.columns:
                    # Get values
                    current_values = df[current_col]
                    baseline_values = df[baseline_col]
                    
                    # Filter for valid data
                    valid_indices = current_values.notna() & baseline_values.notna()
                    
                    if valid_indices.sum() > 0:
                        # Calculate difference (baseline - current)
                        diff_values = (baseline_values[valid_indices] - current_values[valid_indices]).tolist()
                        combined_ring_data[ring_num].extend(diff_values)
        
        # Skip if insufficient data
        if len(combined_ring_data) <= 1:
            ax.text(0.5, 0.5, f"Insufficient data for combined regions", 
                    ha='center', va='center', transform=ax.transAxes)
            return {'results': {'status': 'insufficient_data'}, 'figure': fig}
        
        # Prepare data for JT test
        groups_for_jt = [combined_ring_data[ring] for ring in ring_numbers if ring in combined_ring_data]
        ring_labels = [ring for ring in ring_numbers if ring in combined_ring_data]
        
        # Convert to R format
        r_data = [ro.FloatVector(group) for group in groups_for_jt]
        r_list = ro.ListVector(dict(zip([str(ring) for ring in ring_labels], r_data)))
        
        # Map alternative
        r_alternative = 'greater' if alternative == 'increasing' else 'less' if alternative == 'decreasing' else 'two.sided'
        
        # Run JT test
        jt_result_r = pmcmr.jonckheereTest(r_list, alternative=r_alternative)
        p_value = jt_result_r.rx2('p.value')[0]
        statistic = jt_result_r.rx2('statistic')[0]
        
        # Store results
        results['combined'] = {
            'p_value': p_value,
            'statistic': statistic,
            'alternative': alternative
        }
        
        # Plot mean values by ring
        mean_values = [np.mean(combined_ring_data[r]) if r in combined_ring_data else np.nan for r in ring_numbers]
        
        # Line plot of means
        ax.plot(ring_numbers, mean_values, marker='o', linestyle='-', 
                color=colors['combined'], label=f"Combined (p={p_value:.4f})")
        
        # Add error bars
        y_err = [np.std(combined_ring_data[r])/np.sqrt(len(combined_ring_data[r])) if r in combined_ring_data else np.nan 
                for r in ring_numbers]
        ax.errorbar(ring_numbers, mean_values, yerr=y_err, fmt='none', 
                    ecolor=colors['combined'], alpha=0.5)
        
    else:
        # Process each region separately
        for curr_region in regions_to_analyze:
            # Data structures to store FA differences by ring
            ring_data = defaultdict(list)
            
            # Calculate difference for each ring within specified range
            for ring_num in ring_numbers:
                current_col = f"{parameter}_{curr_region}_ring_{ring_num}"
                baseline_col = f"{parameter}_baseline_{curr_region}_ring_{ring_num}"
                
                if current_col in df.columns and baseline_col in df.columns:
                    # Get values
                    current_values = df[current_col]
                    baseline_values = df[baseline_col]
                    
                    # Filter for valid data
                    valid_indices = current_values.notna() & baseline_values.notna()
                    
                    if valid_indices.sum() > 0:
                        # Calculate difference (baseline - current)
                        diff_values = (baseline_values[valid_indices] - current_values[valid_indices]).tolist()
                        ring_data[ring_num] = diff_values
            
            # Skip region if insufficient data
            if len(ring_data) <= 1:
                results[curr_region] = {
                    'status': 'insufficient_data',
                    'message': f"Insufficient data for {curr_region} region"
                }
                continue
                
            # Prepare data for Jonckheere-Terpstra test
            groups_for_jt = [ring_data[ring] for ring in ring_numbers if ring in ring_data]
            ring_labels = [ring for ring in ring_numbers if ring in ring_data]
            
            # Convert to R format
            r_data = [ro.FloatVector(group) for group in groups_for_jt]
            r_list = ro.ListVector(dict(zip([str(ring) for ring in ring_labels], r_data)))
            
            # Map Python alternative to R alternative
            r_alternative = 'greater' if alternative == 'increasing' else 'less' if alternative == 'decreasing' else 'two.sided'
            
            # Run the Jonckheere-Terpstra test using R's PMCMRplus package
            jt_result_r = pmcmr.jonckheereTest(r_list, alternative=r_alternative)
            
            # Extract results from R output
            p_value = jt_result_r.rx2('p.value')[0]
            statistic = jt_result_r.rx2('statistic')[0]
            
            # Create dictionary with test results
            jt_result = {
                'statistic': statistic,
                'p_value': p_value,
                'alternative': alternative
            }
            
            # Store results
            results[curr_region] = jt_result
            
            # Plot mean values by ring
            mean_values = [np.mean(ring_data[r]) if r in ring_data else np.nan for r in ring_numbers]
            
            # Line plot of means
            ax.plot(ring_numbers, mean_values, marker='o', linestyle='-', 
                   color=colors[curr_region], label=f"{curr_region.capitalize()} (p={p_value:.4f})")
            
            # Add error bars (standard error)
            y_err = [np.std(ring_data[r])/np.sqrt(len(ring_data[r])) if r in ring_data else np.nan 
                    for r in ring_numbers]
            ax.errorbar(ring_numbers, mean_values, yerr=y_err, fmt='none', 
                       ecolor=colors[curr_region], alpha=0.5)
    
    # Add plot elements
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Ring Number", fontsize=14)
    ax.set_ylabel(f"{parameter.upper()} Difference (Control - Craniectomy)", fontsize=14)
    ax.set_xticks(ring_numbers)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    title_prefix = "Combined" if combine_regions else "Separate"
    plt.title(f"{title_prefix} Regions: Jonckheere-Terpstra Test for White Matter {parameter.upper()} Differences Across Rings {start_ring}-{end_ring}", 
             fontsize=16)
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #get basename
        basename=os.path.basename(save_path)
        thesis_path=f"../Thesis/phd-thesis-template-2.4/Chapter6/Figs/{basename}"
        # save to ../Thesis/phd-thesis-template-2.4/Chapter6/Figs/<basename>
        plt.savefig(thesis_path, dpi=600, bbox_inches='tight')
        print(f"Figure saved to {save_path} and {thesis_path}")
    
    # Return results and figure
    return {'results': results, 'figure': fig}


def plot_metric_difference(df, parameter, region, save_path=None, plot_type='box', group_by='region'):
    """
    Plot the difference between baseline and current values for a specific parameter and region across ROIs.
    
    Args:
        df: DataFrame with the data
        parameter: The metric to plot (e.g., 'fa', 'md')
        region: Region to analyze ('anterior', 'posterior', or 'both')
        save_path: Path to save the figure (optional)
        plot_type: Type of plot ('box', 'violin', 'scatter', or 'strip')
        group_by: How to group the data ('region' or 'timepoint')
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import re
    import seaborn as sns
    
    # Set publication style (assuming this function exists in your environment)
    try:
        set_publication_style()
    except NameError:
        plt.style.use('seaborn-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.figsize': (12, 8)
        })
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique timepoints
    timepoints = df['timepoint'].unique()  
    
    # Define the desired order for timepoints
    timepoint_order = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']

    # Filter to only include timepoints present in your data
    present_timepoints = [tp for tp in timepoint_order if tp in timepoints]
        
    # Create color map for timepoints based on the ordered list
    cmap = plt.cm.get_cmap('viridis', len(present_timepoints))
    timepoint_colors = {tp: cmap(i) for i, tp in enumerate(present_timepoints)}
    
    # Determine regions to analyze
    regions_to_analyze = []
    if region == 'both':
        regions_to_analyze = ['anterior', 'posterior']
    else:
        regions_to_analyze = [region]
    
    # Create an empty DataFrame to store the difference data
    diff_df = pd.DataFrame()


    
    # Process each region
    for curr_region in regions_to_analyze:
        # Find all columns matching the parameter and current region
        parameter = parameter.lower()
        current_pattern = f"{parameter}_{curr_region}_ring_(\\d+)"
        baseline_pattern = f"{parameter}_baseline_{curr_region}_ring_(\\d+)"
        
        current_cols = [col for col in df.columns if re.match(current_pattern, col.lower())]
        baseline_cols = [col for col in df.columns if re.match(baseline_pattern, col.lower())]
        
        if not current_cols or not baseline_cols:
            continue
        
        # Extract ring numbers and create mapping between current and baseline columns
        ring_mapping = {}
        ring_numbers = []
        
        for col in current_cols:
            match = re.search(r'_ring_(\d+)$', col)
            if match:
                ring_num = int(match.group(1))
                ring_numbers.append(ring_num)
                baseline_col = f"{parameter}_baseline_{curr_region}_ring_{ring_num}"
                if baseline_col in baseline_cols:
                    ring_mapping[col] = baseline_col
        
        # Sort unique ring numbers
        unique_ring_numbers = sorted(set(ring_numbers))
        
        # Calculate differences for each timepoint and ring
        for tp in timepoints:
            tp_data = df[df['timepoint'] == tp]
            
            for current_col, baseline_col in ring_mapping.items():
                # Extract ring number
                match = re.search(r'_ring_(\d+)$', current_col)
                if not match:
                    continue
                    
                ring_num = int(match.group(1))
                
                # Get the difference (baseline - current)
                current_values = tp_data[current_col]
                baseline_values = tp_data[baseline_col]
                
                # Filter for rows where both values are not NaN
                valid_indices = current_values.notna() & baseline_values.notna()
                
                if valid_indices.sum() > 0:
                    # Calculate difference
                    diff_values = baseline_values[valid_indices] - current_values[valid_indices]
                    
                    # Create a temporary DataFrame for this difference
                    temp_df = pd.DataFrame({
                        'difference': diff_values,
                        'ring': [f"Ring {ring_num}"] * len(diff_values),
                        'timepoint': [tp] * len(diff_values),
                        'region': [curr_region.capitalize()] * len(diff_values)
                    })
                    
                    # Append to the main difference DataFrame
                    diff_df = pd.concat([diff_df, temp_df])
    
    # Check if we have data to plot
    if len(diff_df) == 0:
        ax.text(0.5, 0.5, f"No matching data found for {parameter.upper()} in the specified region(s)", 
                ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    # After creating diff_df but before plotting
    diff_df['timepoint'] = pd.Categorical(
        diff_df['timepoint'], 
        categories=present_timepoints,
        ordered=True
    )
    
    # Create the appropriate plot based on plot_type and group_by
    if plot_type == 'box':
        # Create a box plot
        if group_by == 'region' and len(regions_to_analyze) > 1:
            # Group by region (anterior vs posterior)
            sns.boxplot(
                data=diff_df,
                x='ring', y='difference', hue='region',
                palette='Set1',
                ax=ax
            )
            
            # Add points on top of the boxes
            sns.stripplot(
                data=diff_df,
                x='ring', y='difference', hue='region',
                palette='Set1',
                dodge=True,
                alpha=0.7,
                size=4,
                edgecolor='black',
                linewidth=0.5,
                ax=ax,
                legend=False
            )
        else:
            # Group by timepoint
            sns.boxplot(
                data=diff_df,
                x='ring', y='difference', hue='timepoint',
                palette=timepoint_colors,
                ax=ax
            )
            
            # Add points on top of the boxes
            sns.stripplot(
                data=diff_df,
                x='ring', y='difference', hue='timepoint',
                palette=timepoint_colors,
                dodge=True,
                alpha=0.7,
                size=4,
                edgecolor='black',
                linewidth=0.5,
                ax=ax,
                legend=False
            )
    
    elif plot_type == 'violin':
        # Create a violin plot
        if group_by == 'region' and len(regions_to_analyze) > 1:
            sns.violinplot(
                data=diff_df,
                x='ring', y='difference', hue='region',
                palette='Set1',
                split=True, inner='quartile',
                ax=ax
            )
        else:
            sns.violinplot(
                data=diff_df,
                x='ring', y='difference', hue='timepoint',
                palette=timepoint_colors,
                split=True, inner='quartile',
                ax=ax
            )
    
    elif plot_type == 'scatter':
        # Create a scatter plot with jitter
        if group_by == 'region' and len(regions_to_analyze) > 1:
            # Use region for color
            for region_name in diff_df['region'].unique():
                region_data = diff_df[diff_df['region'] == region_name]
                
                # Get unique rings
                rings = region_data['ring'].unique()
                
                for i, ring in enumerate(sorted(rings)):
                    ring_data = region_data[region_data['ring'] == ring]
                    
                    # Create jitter
                    x = np.full(len(ring_data), i)
                    x = x + np.random.normal(0, 0.05, size=len(x))
                    
                    # Color based on region
                    color = 'C0' if region_name == 'Anterior' else 'C1'
                    
                    # Plot with slight jitter on x-axis
                    ax.scatter(
                        x, ring_data['difference'], 
                        color=color,
                        s=50, alpha=0.7, edgecolors='black', linewidths=0.5,
                        label=region_name if i == 0 else ""  # Only add to legend once
                    )
        else:
            # Use timepoint for color
            for tp in timepoints:
                tp_data = diff_df[diff_df['timepoint'] == tp]
                
                if len(tp_data) > 0:
                    # Add jitter to the categorical x positions
                    rings = tp_data['ring'].unique()
                    for i, ring in enumerate(sorted(rings)):
                        ring_data = tp_data[tp_data['ring'] == ring]
                        
                        # Create jitter
                        x = np.full(len(ring_data), i)
                        x = x + np.random.normal(0, 0.05, size=len(x))
                        
                        # Plot with slight jitter on x-axis
                        ax.scatter(
                            x, ring_data['difference'], 
                            color=timepoint_colors[tp],
                            s=50, alpha=0.7, edgecolors='black', linewidths=0.5,
                            label=tp if i == 0 else ""  # Only add to legend once
                        )
        
        # Set x-ticks
        ax.set_xticks(range(len(diff_df['ring'].unique())))
        ax.set_xticklabels(sorted(diff_df['ring'].unique()))
        
    elif plot_type == 'strip':
        # Create a strip plot
        if group_by == 'region' and len(regions_to_analyze) > 1:
            sns.stripplot(
                data=diff_df,
                x='ring', y='difference', hue='region',
                palette='Set1',
                dodge=True,
                alpha=0.7,
                size=8,
                edgecolor='black',
                linewidth=0.5,
                ax=ax
            )
        else:
            sns.boxplot(
                data=diff_df,
                x='ring', y='difference',
                color='lightgray',
                ax=ax,
                showfliers=False
            )
            sns.stripplot(
                data=diff_df,
                x='ring', y='difference', hue='timepoint',
                palette={tp: timepoint_colors[tp] for tp in timepoint_order if tp in diff_df['timepoint'].unique()},
                dodge=True,
                alpha=0.7,
                size=8,
                edgecolor='black',
                linewidth=0.5,
                ax=ax
            )
    
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    if group_by == 'region' and len(regions_to_analyze) > 1:
        ax.legend(by_label.values(), by_label.keys(), title='Region')
    else:
        ax.legend(by_label.values(), by_label.keys(), title='Timepoint')
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Set axis labels and title
    ax.set_xlabel('Ring Number')
    ax.set_ylabel(f'{parameter.upper()} Difference (Control Side - Craniectomy Side)')
    
    # Set appropriate y-limits based on parameter
    if parameter.lower() == 'fa':
        # FA values typically range from 0 to 1, differences will be smaller
        ax.set_ylim(-0.3, 0.3)
    elif parameter.lower() == 'md':
        # MD values are typically very small (e.g., 0.0001 to 0.003)
        ax.set_ylim(-0.001, 0.001)
    
    # Format region name for title
    if region == 'both':
        region_title = "Anterior and Posterior Regions"
    else:
        region_title = f"{region.capitalize()} Region"
    
    # Format group name for title
    if group_by == 'region' and len(regions_to_analyze) > 1:
        group_title = "by Region"
    else:
        group_title = "by Timepoint"
        
    ax.set_title(f'{parameter.upper()} Differences (Control Side - Craniectomy Side) for {region_title} {group_title}')
    
    if 'wm' in save_path:
        ax.set_title(f'{parameter.upper()} Differences (Control Side - Craniectomy Side) in White Matter for {region_title} {group_title}')


    # Adjust layout
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # Get basename from e.g. DTI_Processing_Scripts/test_results/roi_md_wm_10x4vox_both_regions_comparison_boxplot.png
        # replace .png with .pdf and set dpi to 300
        basename = os.path.basename(save_path)
        basename = basename.replace('.png', '.pdf')
        plt.savefig(f'../Thesis/phd-thesis-template-2.4/Chapter6/Figs/{basename}', dpi=300, bbox_inches='tight')
        
    return fig, ax

def create_hex_color_map_from_cmap(cmap_name, n):
    """Create a list of hex colors from a colormap."""
    cmap = plt.get_cmap(cmap_name)
    # Convert the colormap to a list of hex colors
    colors = cmap(np.linspace(0, 1, n))
    hex_colors = [rgb2hex(color) for color in colors]
    return hex_colors

def create_hex_color_map_custom(base_colors, n):
    """Create a list of hex colors from a list of base colors."""
    # Create a custom colormap with the base colors
    cmap = LinearSegmentedColormap.from_list('custom', base_colors, N=n)
    # Convert the colormap to a list of hex colors
    colors = cmap(np.linspace(0, 1, n))
    hex_colors = [rgb2hex(color) for color in colors]
    return hex_colors

def get_color(patient_id, color_map):
    """
    Get color for a patient ID from color_map. If patient ID is not in 
    color_map, dynamically assign a new color and add it to color_map.
    """
    pid = str(patient_id)
    
    # First check if this pid is already in the color map (as string)
    if pid in color_map:
        return color_map[pid]
    
    # If we get here, this patient ID needs a new color
    # Predefined colors for variety
    predefined_colors = [
        'red', 'blue', 'green', 'orange', 'purple', 
        'brown', 'pink', 'olive', 'cyan', 'magenta',
        'gold', 'limegreen', 'darkviolet', 'deepskyblue', 'crimson',
        'darkgreen', 'darkblue', 'darkorange', 'hotpink', 'teal'
    ]
    
    if len(color_map) < len(predefined_colors):
        new_color = predefined_colors[len(color_map)]
    else:
        # Use tab10 colormap for additional colors
        color_idx = len(color_map) % 10
        new_color = plt.cm.tab10(color_idx)
    
    # Add the new color to the color_map for future use
    color_map[pid] = new_color
    
    # Print for debugging
    print(f"Assigned new color to patient {pid}: {new_color}")
    
    return new_color

def plot_all_rings_combined(df, parameter, num_bins=5, save_path=None):
    """
    Plot all rings data on a single figure with days since injury on x-axis.
    Each patient has a unique color, and different ring types use different markers/line styles.
    
    Args:
        df: DataFrame with the data
        save_path: Path to save the figure (optional)
    """
    set_publication_style()
    
    # Get unique patient IDs
    patient_ids = df['patient_id'].unique()
    
    # Create color map for patients
    patient_color_map = {}
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(15, 10))
    fig2, ax2 = plt.subplots(figsize=(15, 10))  # Timepoint category plot
    
    # Define markers for different ring types to distinguish them
    # Define markers for different ring types to distinguish them (without parameter prefixes)
    
    if num_bins == 5:
        ring_markers = {
            'anterior_ring_1': 'o',
            'anterior_ring_2': 's',
            'anterior_ring_3': '^',
            'anterior_ring_4': 'D',
            'anterior_ring_5': 'p',
            'posterior_ring_1': 'o',
            'posterior_ring_2': 's',
            'posterior_ring_3': '^',
            'posterior_ring_4': 'D',
            'posterior_ring_5': 'p',
        }
    elif num_bins == 10:
        ring_markers = {
            'anterior_ring_1': 'o',
            'anterior_ring_2': 's',
            'anterior_ring_3': '^',
            'anterior_ring_4': 'D',
            'anterior_ring_5': 'p',
            'anterior_ring_6': 'X',
            'anterior_ring_7': 'H',
            'anterior_ring_8': 'P',
            'anterior_ring_9': 'v',
            'anterior_ring_10': '<',
            'posterior_ring_1': 'o',
            'posterior_ring_2': 's',
            'posterior_ring_3': '^',
            'posterior_ring_4': 'D',
            'posterior_ring_5': 'p',
            'posterior_ring_6': 'X',
            'posterior_ring_7': 'H',
            'posterior_ring_8': 'P',
            'posterior_ring_9': 'v',
            'posterior_ring_10': '<',
        }
    
    
    # Define line styles for anterior vs posterior
    line_styles = {
        'anterior': '-',
        'posterior': '--',
    }
    
    # Define alpha values for baseline vs current
    alpha_values = {
        'baseline': 0.4,
        'current': 0.9,
    }
    
    # For legend
    legend_handles = []
    legend_labels = []
    patient_added_to_legend = set()
    ring_type_added_to_legend = set()

    # fig2 plotting
    # Define timepoint order 
    timepoint_order = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
    
    # Create a mapping for timepoints to numerical positions
    timepoint_spacing = 3.0  # Adjust this value to control the gutter width
    # timepoint_positions = {tp: i * timepoint_spacing for i, tp in enumerate(timepoint_order)}

    # We'll use a simple numeric sequence as the base positions
    base_positions = list(range(len(timepoint_order)))
    
    # Apply spacing to create the actual positions
    spaced_positions = [pos * timepoint_spacing for pos in base_positions]
    
    # Create mapping from timepoint names to spaced positions
    timepoint_positions = {tp: spaced_positions[i] for i, tp in enumerate(timepoint_order)}

    # Create a numerical index for each patient ID to determine offset
    patient_indices = {pid: i for i, pid in enumerate(sorted(patient_ids))}
    total_patients = len(patient_ids)

    # Calculate the width to allocate for each timepoint
    timepoint_width = 0.8  # Width allocated for each timepoint
    
    # All possible column names for rings
    ring_cols = []
    for region in ['anterior', 'posterior']:
        for ring_num in range(1, num_bins + 1):
            ring_cols.append(f'{parameter}_{region}_ring_{ring_num}')
            ring_cols.append(f'{parameter}_baseline_{region}_ring_{ring_num}')
    
    #print(f"Ring columns: {ring_cols}")
    
    # For the timepoint plot, first collect all data points per timepoint and patient
    # to determine better offset calculations
    timepoint_data_collection = {}
    
    # First pass: collect data by timepoint and patient
    for pid in patient_ids:
        patient_data = df[df['patient_id'] == pid].copy()
        if patient_data.empty or 'timepoint' not in patient_data.columns:
            continue
            
        for tp in timepoint_order:
            if tp not in timepoint_data_collection:
                timepoint_data_collection[tp] = {}
                
            tp_data = patient_data[patient_data['timepoint'] == tp]
            if not tp_data.empty:
                timepoint_data_collection[tp][pid] = tp_data
    
    # For each timepoint, determine which patients have data and assign positions
    timepoint_patient_positions = {}
    for tp in timepoint_order:
        if tp in timepoint_data_collection:
            # Get patients that have data for this timepoint
            patients_with_data = list(timepoint_data_collection[tp].keys())
            
            # If there are patients with data
            if patients_with_data:
                # Sort patients for consistent ordering
                patients_with_data.sort()
                
                # Create evenly distributed positions for this timepoint
                timepoint_width = 0.8  # Width allocated for each timepoint
                
                # Assign positions
                if len(patients_with_data) > 1:
                    for i, pid in enumerate(patients_with_data):
                        # Calculate offset to center the distribution
                        offset = (i - (len(patients_with_data) - 1) / 2) * (timepoint_width / len(patients_with_data))
                        
                        if tp not in timepoint_patient_positions:
                            timepoint_patient_positions[tp] = {}
                        timepoint_patient_positions[tp][pid] = offset
                else:
                    # Just one patient, no offset needed
                    if tp not in timepoint_patient_positions:
                        timepoint_patient_positions[tp] = {}
                    timepoint_patient_positions[tp][patients_with_data[0]] = 0




    # Loop through patients
    for pid in patient_ids:
        patient_data = df[df['patient_id'] == pid].copy()
        # Skip patients with no data
        if patient_data.empty:
            continue
            
        # Sort by days since injury
        patient_data = patient_data.sort_values('Days_since_injury')
        
        # Get color for this patient
        patient_color = get_color(pid, patient_color_map)

        # Calculate patient-specific offset within timepoint
        if total_patients > 1:
            # Scale to fit within the timepoint width
            patient_offset = (patient_indices[pid] - (total_patients - 1) / 2) * (timepoint_width / total_patients)
        else:
            patient_offset = 0
        
        # Loop through all ring columns
        for col in ring_cols:
            if col not in patient_data.columns:
                continue
                
            # Skip columns with all NaN values
            if patient_data[col].isna().all():
                continue
                
            # Determine if this is baseline or current data
            is_baseline = 'baseline' in col
            
            # Determine region (anterior or posterior)
            region = 'anterior' if 'anterior' in col else 'posterior'
            
            # Extract ring number
            ring_num = int(col.split('_')[-1])
            
            # Determine marker and line style
            ring_key = f"{region}_ring_{ring_num}"
            marker = ring_markers[ring_key]
            line_style = line_styles[region]
            alpha = alpha_values['baseline' if is_baseline else 'current']
            
            # Filter out NaN values
            valid_data = patient_data[~patient_data[col].isna()]
            if len(valid_data) == 0:
                continue
                
            # Plot scatter points
            scatter = ax.scatter(
                valid_data['Days_since_injury'], 
                valid_data[col],
                color=patient_color,
                marker=marker,
                s=30 if is_baseline else 50,
                alpha=alpha,
                edgecolors='black' if not is_baseline else None,
                linewidths=0.5 if not is_baseline else 0
            )

            # PLOT 2: Timepoints (discrete)
            # Only plot if timepoint data is available

            # PLOT 2: Timepoints (discrete) with improved patient-specific offset
            if 'timepoint' in valid_data.columns:
                for _, row in valid_data.iterrows():
                    timepoint = row['timepoint']
                    if pd.notna(timepoint) and timepoint in timepoint_positions:
                        # Use the pre-calculated patient position for this timepoint if available
                        if (timepoint in timepoint_patient_positions and 
                            pid in timepoint_patient_positions[timepoint]):
                            patient_offset = timepoint_patient_positions[timepoint][pid]
                        else:
                            patient_offset = 0  # Fallback if position not calculated
                            
                        # Base position + patient-specific offset
                        x_pos = timepoint_positions[timepoint] + patient_offset
                        
                        ax2.scatter(
                            x_pos,
                            row[col],
                            color=patient_color,
                            marker=marker,
                            s=30 if is_baseline else 50,
                            alpha=alpha,
                            edgecolors='black' if not is_baseline else None,
                            linewidths=0.5 if not is_baseline else 0
                        )
            # if 'timepoint' in valid_data.columns:
            #     # Convert timepoints to numeric positions
            #     valid_data['timepoint_pos'] = valid_data['timepoint'].map(timepoint_positions)
                
            #     # Only use data points with valid timepoints
            #     timepoint_data = valid_data.dropna(subset=['timepoint_pos'])
                
            #     if not timepoint_data.empty:
            #         ax2.scatter(
            #             timepoint_data['timepoint_pos'],
            #             timepoint_data[col],
            #             color=patient_color,
            #             marker=marker,
            #             s=30 if is_baseline else 50,
            #             alpha=alpha,
            #             edgecolors='black' if not is_baseline else None,
            #             linewidths=0.5 if not is_baseline else 0
            #         )

            
            # Add to legend (only add each patient and ring type once)
            if pid not in patient_added_to_legend:
                legend_handles.append(Line2D([0], [0], color=patient_color, marker='o', linestyle='-', 
                                           markersize=6, label=f'Patient {pid}'))
                legend_labels.append(f'Patient {pid}')
                patient_added_to_legend.add(pid)
            
            # Create a unique identifier for this ring type
            ring_type_id = f"{region}_ring_{ring_num}"
            if ring_type_id not in ring_type_added_to_legend:
                legend_handles.append(Line2D([0], [0], color='black', marker=marker, linestyle=line_style, 
                                           markersize=6))
                legend_labels.append(f"{region.capitalize()} Ring {ring_num}")
                ring_type_added_to_legend.add(ring_type_id)
            
            # If we have enough points (>=2), plot a smooth curve
            if not is_baseline and len(valid_data) >= 3:
                x_data = valid_data['Days_since_injury'].values
                y_data = valid_data[col].values
                
                try:
                    # Create smooth x values for the curve
                    x_smooth = np.linspace(min(x_data), max(x_data), 100)
                    
                    # Use cubic spline for smoothing
                    cs = CubicSpline(x_data, y_data, bc_type='natural')
                    y_smooth = cs(x_smooth)
                    
                    # Plot the smooth curve
                    # ax.plot(x_smooth, y_smooth, color=patient_color, alpha=0.7, 
                    #        linestyle=line_style, linewidth=1.0)
                except Exception as e:
                    print(f"Could not create smooth curve for patient {pid}, {col}: {e}")
    
    # Add any missing ring types to the legend when num_bins=10
    if num_bins == 10:
        for region in ['anterior', 'posterior']:
            for ring_num in range(1, 11):
                ring_type_id = f"{region}_ring_{ring_num}"
                if ring_type_id not in ring_type_added_to_legend:
                    marker = ring_markers[ring_type_id]
                    line_style = line_styles[region]
                    legend_handles.append(Line2D([0], [0], color='black', marker=marker, linestyle=line_style, 
                                            markersize=6))
                    legend_labels.append(f"{region.capitalize()} Ring {ring_num}")
                    ring_type_added_to_legend.add(ring_type_id)


    # Add baseline vs current markers to legend
    legend_handles.append(plt.Line2D([0], [0], color='black', marker='o', linestyle='None', 
                               markersize=6, alpha=alpha_values['current'], 
                               markeredgecolor='black', markeredgewidth=0.5))
    legend_labels.append('Current')
    
    legend_handles.append(plt.Line2D([0], [0], color='black', marker='o', linestyle='None', 
                               markersize=6, alpha=alpha_values['baseline']))
    legend_labels.append('Baseline')
    
    # # Add anterior vs posterior line styles to legend
    # legend_handles.append(Line2D([0], [0], color='black', linestyle=line_styles['anterior']))
    # legend_labels.append('Anterior')
    
    # legend_handles.append(Line2D([0], [0], color='black', linestyle=line_styles['posterior']))
    # legend_labels.append('Posterior')
    
    # Set labels and title
    ax.set_xlabel('Days Since Injury')
    ax.set_ylabel('Fractional Anisotropy (FA)')
    ax.set_title('Fractional Anisotropy: All Rings Over Time')
    
    
    # Set x-axis to start at 0 and find the maximum value for upper limit
    max_days = 0
    for pid in patient_ids:
        patient_data = df[df['patient_id'] == pid]
        if not patient_data.empty and 'Days_since_injury' in patient_data.columns:
            patient_max = patient_data['Days_since_injury'].max()
            if not pd.isna(patient_max) and patient_max > max_days:
                max_days = patient_max
    
    # Add a small margin on the right (5%)
    margin = 0.05 * max_days
    ax.set_xlim(0, max_days + margin)
    if 'fa' in parameter:
        ax.set_ylim(0, 0.5)
    else:
        ax.set_ylim(0, 0.0035)
    


    # Set labels and title for Plot 2 (Timepoint category)
    ax2.set_xlabel('Timepoint')
    ax2.set_ylabel(f'{parameter.upper()}')
    ax2.set_title(f'{parameter.upper()}: All Rings Over Time (Discrete Timepoints)')
    
    # Set x-ticks for Plot 2 to be the timepoint names
    ax2.set_xticks(spaced_positions)
    ax2.set_xticklabels(timepoint_order, rotation=45)
    # Set x-axis limits for Plot 2 to include some padding
    # min_pos = min(timepoint_positions.values()) - 0.5 * timepoint_spacing
    # max_pos = max(timepoint_positions.values()) + 0.5 * timepoint_spacing
    ax2.set_xlim(-0.5, spaced_positions[-1] + 0.5)
    if 'fa' in parameter:
        ax.set_ylim(0, 0.5)
    else:
        ax.set_ylim(0, 0.0035)
    

        # Add vertical lines at each timepoint to further separate them visually
    for pos in spaced_positions:
        ax2.axvline(x=pos - 0.4, color='lightgray', linestyle='-', alpha=0.3)
        ax2.axvline(x=pos + 0.4, color='lightgray', linestyle='-', alpha=0.3)
    
    # Add legend to both plots
    for ax in [ax, ax2]:
        ax.legend(handles=legend_handles, labels=legend_labels, loc='center left', 
                 bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1)
    
    # Adjust layout for both plots
    for figure in [fig, fig2]:
        figure.tight_layout()
        plt.figure(figure.number)
        plt.subplots_adjust(right=0.8)
    
    # Save figures if path is provided
    if save_path:
        # Create file paths by inserting a suffix before the extension
        base, ext = os.path.splitext(save_path)
        days_path = f"{base}_days{ext}"
        timepoint_path = f"{base}_timepoint{ext}"
        
        fig.savefig(days_path, bbox_inches='tight')
        fig2.savefig(timepoint_path, bbox_inches='tight')
        print(f"Figures saved to {days_path} and {timepoint_path}")
    
    return (fig, ax), (fig2, ax2)

def plot_metric_roi(df, parameter, region_type, save_path=None, plot_type='scatter'):
    """
    Plot the metric for a specific region type across all patients.
    
    Args:
        df: DataFrame with the data
        parameter: The metric to plot (e.g., 'fa', 'md')
        region_type: Type of region ('anterior', 'posterior', 'baseline_anterior', 'baseline_posterior')
        save_path: Path to save the figure (optional)
        plot_type: Type of plot ('scatter', 'box', or 'violin')
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import re
    import seaborn as sns
    
    # Set publication style
    set_publication_style()
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get unique timepoints
    timepoints = df['timepoint'].unique()  
    
    # Define the desired order for timepoints
    timepoint_order = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']

    # Filter to only include timepoints present in your data
    present_timepoints = [tp for tp in timepoint_order if tp in timepoints]
        
    # Create color map for timepoints based on the ordered list
    cmap = plt.cm.get_cmap('viridis', len(present_timepoints))
    timepoint_colors = {tp: cmap(i) for i, tp in enumerate(present_timepoints)}
    
    # Find all columns matching the parameter and region type
    parameter = parameter.lower()
    pattern = f"{parameter}_{region_type}_ring_(\\d+)"
    matching_cols = [col for col in df.columns if re.match(pattern, col.lower())]
    
    if not matching_cols:
        ax.text(0.5, 0.5, f"No data found for {parameter.upper()} in {region_type} rings", 
               ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    # Extract ring numbers and sort columns by ring number
    ring_numbers = []
    for col in matching_cols:
        match = re.search(r'_ring_(\d+)$', col)
        if match:
            ring_numbers.append(int(match.group(1)))
        else:
            ring_numbers.append(0)
            
    # Sort columns by ring number
    sorted_indices = np.argsort(ring_numbers)
    sorted_cols = [matching_cols[i] for i in sorted_indices]
    sorted_ring_numbers = [ring_numbers[i] for i in sorted_indices]
    
    # Prepare data for plotting
    plot_df = pd.DataFrame()
    
    for i, col in enumerate(sorted_cols):
        ring_num = sorted_ring_numbers[i]
        
        for tp in timepoints:
            tp_data = df[df['timepoint'] == tp]
            values = tp_data[col].dropna()
            
            if len(values) > 0:
                # Create a temporary DataFrame for this ring/timepoint combination
                temp_df = pd.DataFrame({
                    'value': values,
                    'ring': [f"Ring {ring_num}"] * len(values),
                    'timepoint': [tp] * len(values)
                })
                
                # Append to the main plotting DataFrame
                plot_df = pd.concat([plot_df, temp_df])
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f"No data found for {parameter.upper()} in {region_type} rings", 
               ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    # Create appropriate plot based on plot_type
    if plot_type == 'scatter':
        # Create a scatter plot with jitter
        for tp in timepoints:
            tp_data = plot_df[plot_df['timepoint'] == tp]
            
            if len(tp_data) > 0:
                # Add jitter to the categorical x positions
                rings = tp_data['ring'].unique()
                for i, ring in enumerate(sorted(rings)):
                    ring_data = tp_data[tp_data['ring'] == ring]
                    
                    # Create jitter
                    x = np.full(len(ring_data), i)
                    x = x + np.random.normal(0, 0.05, size=len(x))
                    
                    # Plot with slight jitter on x-axis
                    ax.scatter(
                        x, ring_data['value'], 
                        color=timepoint_colors[tp],
                        s=50, alpha=0.7, edgecolors='black', linewidths=0.5,
                        label=tp if i == 0 else ""  # Only add to legend once
                    )
        
        # Set x-ticks
        ax.set_xticks(range(len(sorted_ring_numbers)))
        ax.set_xticklabels([f"Ring {ring}" for ring in sorted_ring_numbers])
        
    elif plot_type == 'box':
        # Create a box plot
        sns.boxplot(
            data=plot_df,
            x='ring', y='value', hue='timepoint',
            palette=timepoint_colors,
            ax=ax
        )

        # Add strip plot on top (this automatically adds dots over the boxes)
        sns.stripplot(
            data=plot_df,
            x='ring', y='value', hue='timepoint',
            palette=timepoint_colors,
            dodge=True,  # This makes the points align with their respective boxes
            alpha=0.7,
            size=4,
            edgecolor='black',
            linewidth=0.5,
            ax=ax,
            legend=False  # Don't add a second legend
        )
    elif plot_type == 'strip':
        # one box plot per thingy
        sns.boxplot(
            data=plot_df,
            x='ring', y='value',
            color='lightgray',  # Plain gray boxes
            ax=ax,
            showfliers=False
        )
        # Add strip plot on top (this automatically adds dots over the boxes)
        sns.stripplot(
            data=plot_df,
            x='ring', y='value', hue='timepoint',
            palette=timepoint_colors,
            dodge=True,  # This makes the points align with their respective boxes
            alpha=0.7,
            size=4,
            edgecolor='black',
            linewidth=0.5,
            ax=ax,
            legend=True  # Don't add a second legend
        )
        
    elif plot_type == 'violin':
        # Create a violin plot
        sns.violinplot(
            data=plot_df,
            x='ring', y='value', hue='timepoint',
            palette=timepoint_colors,
            split=True, inner='quartile',
            ax=ax
        )
    
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Timepoint')
    
    # Set axis labels and title
    ax.set_xlabel('Ring Number')
    ax.set_ylabel(f'{parameter.upper()} Value')
    if 'fa' in parameter:
        ax.set_ylim(0, 0.5)
    elif 'md' in parameter:
        ax.set_ylim(0, 0.0035)
    
    # Format region type for title
    region_title = region_type.replace('_', ' ').title()
    if region_title.startswith('Baseline'):
        region_title = region_title.replace('Baseline', 'Control Side')
    elif region_title.startswith('Anterior'):
        region_title = region_title.replace('Anterior', 'Craniectomy Side Anterior')
    elif region_title.startswith('Posterior'):
        region_title = region_title.replace('Posterior', 'Craniectomy Side Posterior')
    ax.set_title(f'{parameter.upper()} Values for {region_title} Rings by Timepoint')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax
    

def process_timepoint_data(input_file_location):
    """
    Process patient timepoint data by standardizing timepoint values and sorting results.
    
    Args:
        df: Input pandas DataFrame containing patient data
        
    Returns:
        Processed pandas DataFrame with standardized timepoints and proper sorting
    """
    # Check if harmonised data file supplied
    if 'harmonised' in input_file_location:
        print('Harmonised data file supplied')
    else:
        print('Harmonised data file not supplied, please check')
        input('Press Enter to continue or Ctrl+C to exit')

        #return None

    df=pd.read_csv(input_file_location)
    df['patient_id'] = df['patient_id'].astype(str)
    timepoints = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
    string_mask = df['timepoint'].isin(timepoints)
    numeric_mask = ~string_mask & df['timepoint'].apply(lambda x: pd.notnull(pd.to_numeric(x, errors='coerce')))
    # Convert numeric values to their appropriate string representations
    for idx in df[numeric_mask].index:
        try:
            numeric_value = float(df.loc[idx, 'timepoint'])
            df.loc[idx, 'timepoint'] = map_timepoint_to_string(numeric_value)
        except (ValueError, TypeError):
            continue
    #df['timepoint'] = df['timepoint'].apply(map_timepoint_to_string) # convert to string category
    #print(f"df:\n{df}")
    # Save original timepoint values before recategorization (if needed for reference)
    df['original_timepoint'] = df['timepoint']
    # Now apply the timepoint recategorization based on Days_since_injury
    # Define the ranges and labels for recategorization
    ranges = [(0, 2), (2, 8), (8, 42), (42, 179), (179, 278), (278, 540), (540, 500000)]
    labels = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
    # Only recategorize rows where Days_since_injury is not null
    mask = df['Days_since_injury'].notnull()
    if mask.any():
        df.loc[mask, 'timepoint'] = pd.cut(
            df.loc[mask, 'Days_since_injury'],
            bins=[0, 2, 8, 42, 179, 278, 540, 500000],
            labels=labels
        )
    # sort data according to patient id then timepoint
    def get_sort_key(patient_id):
        try:
            return (0, int(patient_id)) # Numeric IDs first, sorted numerically
        except ValueError:
            return (1, patient_id) # Alphanumeric IDs second, sorted alphabetically
    # Create a sort key column
    df['sort_key'] = df['patient_id'].apply(get_sort_key)
    df['timepoint_order'] = df['timepoint'].apply(lambda x: timepoints.index(x) if x in timepoints else 999)
    # Sort the dataframe by patient_id, then by the position of timepoint in our list
    df = df.sort_values(by=['sort_key', 'timepoint_order'])
    df = df.drop(['sort_key', 'timepoint_order'], axis=1) # remove sorting column
    
    return df


# Set publication style for matplotlib
set_publication_style()

if __name__ == '__main__':
    print('running dti_results_plotting_main.py')
    
    # Load the (harmonised) data 

    # not harmonised data test
    # data_5x4vox_not_harmonised_filename='DTI_Processing_Scripts/merged_data_5x4vox_NEW_filtered_notharmonised.csv'
    # data_5x4vox_not_harmonised=process_timepoint_data(input_file_location=data_5x4vox_not_harmonised_filename)
    # plot_all_rings_combined(df=data_5x4vox_not_harmonised, parameter='fa', save_path='DTI_Processing_Scripts/test_results/all_rings_combined_5x4vox_not_harmonised.png')
    # plot_all_rings_combined(df=data_5x4vox_not_harmonised, parameter='md', save_path='DTI_Processing_Scripts/test_results/all_rings_combined_5x4vox_not_harmonised_md.png')

    
    # data_5x4vox_filename='DTI_Processing_Scripts/merged_data_5x4vox_NEW_filtered_harmonised.csv'
    # data_5x4vox=process_timepoint_data(input_file_location=data_5x4vox_filename)

    # Now data_5x4vox has been recategorized based on Days_since_injury, exactly the same as the deformation analysis
    # plot_all_rings_combined(df=data_5x4vox, parameter='fa', save_path='DTI_Processing_Scripts/test_results/all_rings_combined_5x4vox_filtered.png')
    # plot_all_rings_combined(df=data_5x4vox, parameter='md', save_path='DTI_Processing_Scripts/test_results/all_rings_combined_5x4vox_filtered_md.png')


    # # Scatter plots for 5x4 vox data
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_anterior_scatter.png')
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_baseline_anterior_scatter.png')
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_posterior_scatter.png')
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_baseline_posterior_scatter.png')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_anterior_scatter.png')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_baseline_anterior_scatter.png')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_posterior_scatter.png')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_baseline_posterior_scatter.png')

    # # Box plots for 5x4 vox data
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_anterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_baseline_anterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_posterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_baseline_posterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_anterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_baseline_anterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_posterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_baseline_posterior_box.png', plot_type='box')

    # # Strip plots for 5x4 vox data
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_anterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_baseline_anterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_posterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_baseline_posterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_anterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_baseline_anterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_posterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_baseline_posterior_strip.png', plot_type='strip')

    # wm_data_5x4vox_filename='DTI_Processing_Scripts/merged_data_5x4vox_NEW_filtered_wm_harmonised.csv'
    # wm_data_5x4vox=process_timepoint_data(input_file_location=wm_data_5x4vox_filename)

    # # Strip plots for 5x4 vox data
    # plot_metric_roi(df=wm_data_5x4vox, parameter='fa', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_wm_5x4vox_anterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=wm_data_5x4vox, parameter='fa', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_wm_5x4vox_baseline_anterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=wm_data_5x4vox, parameter='fa', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_wm_5x4vox_posterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=wm_data_5x4vox, parameter='fa', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_wm_5x4vox_baseline_posterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=wm_data_5x4vox, parameter='md', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_wm_5x4vox_anterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=wm_data_5x4vox, parameter='md', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_wm_5x4vox_baseline_anterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=wm_data_5x4vox, parameter='md', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_wm_5x4vox_posterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=wm_data_5x4vox, parameter='md', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_wm_5x4vox_baseline_posterior_strip.png', plot_type='strip')

    # plot_metric_difference(df=wm_data_5x4vox, parameter='fa', region='anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_wm_5x4vox_anterior_comparison_box.png', plot_type='strip')
    # plot_metric_difference(df=wm_data_5x4vox, parameter='fa', region='both', save_path='DTI_Processing_Scripts/test_results/roi_fa_wm_5x4vox_both_regions_comparison_box.png', plot_type='strip', group_by='timepoint')
    # # plot_metric_difference

    # wm_data_10x4vox_filename='DTI_Processing_Scripts/merged_data_10x4vox_NEW_filtered_wm_harmonised.csv'
    # wm_data_10x4vox=process_timepoint_data(input_file_location=wm_data_10x4vox_filename)
    # # plot_metric_difference(df=wm_data_5x4vox, parameter='fa', region='anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_wm_5x4vox_anterior_comparison_box.png', plot_type='strip')
    # plot_metric_difference(df=wm_data_10x4vox, parameter='fa', region='both', save_path='DTI_Processing_Scripts/test_results/roi_fa_wm_10x4vox_both_regions_comparison_box.png', plot_type='strip', group_by='timepoint')
    # plot_metric_difference(df=wm_data_10x4vox, parameter='md', region='both', save_path='DTI_Processing_Scripts/test_results/roi_md_wm_10x4vox_both_regions_comparison_box.png', plot_type='strip', group_by='timepoint')
    # plot_metric_difference(df=wm_data_10x4vox, parameter='fa', region='both', save_path='DTI_Processing_Scripts/test_results/roi_fa_wm_10x4vox_both_regions_comparison_boxplot.png', plot_type='box', group_by='timepoint')
    # plot_metric_difference(df=wm_data_10x4vox, parameter='md', region='both', save_path='DTI_Processing_Scripts/test_results/roi_md_wm_10x4vox_both_regions_comparison_boxplot.png', plot_type='box', group_by='timepoint')

    # print(wm_data_10x4vox.columns)
    ####################################
    ######### JT TEST #################
    #################################

    # # Run the test on rings 2-10 looking for an increasing trend
    # results = jt_test(df=wm_data_10x4vox, parameter='fa', regions=(2, 10), 
    #                 save_path='DTI_Processing_Scripts/jt_test_results-fa-rings-2to10.png', alternative='increasing')

    # print(results)
    # results = jt_test(df=wm_data_10x4vox, parameter='fa', regions=(2, 10), 
    #                 save_path='DTI_Processing_Scripts/jt_test_results-fa-rings-combined-2to10.png', alternative='increasing', combine_regions=True)

    # print(results)

    # results = jt_test(df=wm_data_10x4vox, parameter='md', regions=(2, 10), 
    #                 save_path='DTI_Processing_Scripts/jt_test_results-md-rings-2to10.png', alternative='increasing')

    # print(results)
    # results = jt_test(df=wm_data_10x4vox, parameter='md', regions=(2, 10), 
    #                 save_path='DTI_Processing_Scripts/jt_test_results-md-rings-combined-2to10.png', alternative='increasing', combine_regions=True)

    # print(results)

    ################################################
    
    #PLOTS FOR ROI - TIMEPOINT BASED
    #################################################

    #Import roi 567 harmonised data
    wm_data_roi_567_filename='DTI_Processing_Scripts/merged_data_10x4vox_NEW_filtered_wm_567_harmonised.csv'
    wm_data_roi_567=process_timepoint_data(input_file_location=wm_data_roi_567_filename)
    # Get differences in fa and md
    wm_data_roi_567=parameter_differences(wm_data_roi_567)
    print(wm_data_roi_567.columns)
    print(wm_data_roi_567.head(1))
    sys.exit()
    
    




    # # Data availability matrix
    # matrix = data_availability_matrix(
    #     data=wm_data_roi_567, 
    #     timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo'],
    #     diff_column='fa_anterior_diff',  # or any other diff column
    #     filename='fa_diff_data_availability.png'
    # )

    # create_timepoint_boxplot_recategorised_dti(df=wm_data_roi_567, parameter='fa', timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo'])
    # create_timepoint_boxplot_recategorised_dti(df=wm_data_roi_567, parameter='md', timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo'])
    
    
    # create_timepoint_boxplot_recategorised_dti_single_region(df=wm_data_roi_567, parameter='fa', region='anterior', timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo'])
    # create_timepoint_boxplot_recategorised_dti_single_region(df=wm_data_roi_567, parameter='md', region='anterior', timepoints=['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo'])

    # # combine 3 and 6 month timepoints, and 12 and 24 mo timepoints.
    wm_data_roi_567_combi=wm_data_roi_567.copy()
    wm_data_roi_567_combi['timepoint']=wm_data_roi_567['timepoint'].replace({
        '3mo': '3-6mo',
        '6mo': '3-6mo',
        '12mo': '12-24mo',
        '24mo': '12-24mo'
    })
    # Remove duplicates, keeping first occurrence (3mo will be kept over 6mo due to original order)
    wm_data_roi_567_combi = wm_data_roi_567_combi.drop_duplicates(subset=['patient_id', 'timepoint'], keep='first')

    # print(f"Unique timepoints in combined wm data: {wm_data_roi_567_combi['timepoint'].unique()}")
    # print(wm_data_roi_567_combi)

    # # Do new data availability matrix for combi data
    # matrix_combi = data_availability_matrix(
    #     data=wm_data_roi_567_combi, 
    #     timepoints=['ultra-fast', 'fast', 'acute', '3-6mo', '12-24mo'],
    #     diff_column='fa_anterior_diff',  # or any other diff column
    #     filename='fa_diff_data_availability_combi.png'
    # )

    #####################################################
    # LINEAR MIXED EFFECTS MODEL WITH COMBI DATA
    # H_0: There is no statistically significant difference between 
    # FA in Control vs. Craniectomy for anterior and posterior ROIs. 
    # i.e. H_0: FA_diff = FA_{control} - FA_{craniectomy} = 0

    # LMER equation with Patient as random effect, Timepoint as fixed effect, 'Region' as a covariate. 
    # Y_{ijk} = \beta_0 + \sum_{t=1}^{T-1} \beta_{1t} \cdot \text{Timepoint}_{jt} + \beta_2 \cdot \text{Region}_k + u_i + \varepsilon_{ijk}

    # $Y_{ijk}$: FA difference (control - craniectomy) for subject $i$, at timepoint $j$, in region $k$  
    # $\beta_0$: Intercept (mean FA difference at reference timepoint and region)  
    # $\beta_{1t}$: Coefficient for timepoint $t$ (excluding the reference level)  
    # $\text{Timepoint}_{jt}$: Indicator variable (1 if observation is at timepoint $t$, else 0)  
    # $\beta_2$: Coefficient for region (e.g., anterior vs posterior)  
    # $\text{Region}_k$: Indicator variable for brain region (e.g., 0 = anterior, 1 = posterior)  
    # $u_i$: Random intercept for subject $i$, where $u_i \sim \mathcal{N}(0, \sigma_u^2)$  
    # $\varepsilon_{ijk}$: Residual error, where $\varepsilon_{ijk} \sim \mathcal{N}(0, \sigma^2)$  

    print(wm_data_roi_567_combi.columns)

    fa_long_wm_data_roi_567_combi = pd.melt(wm_data_roi_567_combi,
                  id_vars=['patient_id', 'timepoint'],
                  value_vars=['fa_anterior_diff', 'fa_posterior_diff'],
                  var_name='Region',
                  value_name='FA_diff')
    
    print(fa_long_wm_data_roi_567_combi)

    # rename regions to anterior and posterior
    fa_long_wm_data_roi_567_combi['Region'] = fa_long_wm_data_roi_567_combi['Region'].map({
        'fa_anterior_diff': 'anterior',
        'fa_posterior_diff': 'posterior'
    })

    # Order timepoints, so that the first one is reference
    # Ensure 'timepoint' is treated as a categorical variable
    fa_long_wm_data_roi_567_combi['timepoint'] = pd.Categorical(
        fa_long_wm_data_roi_567_combi['timepoint'],
        categories=['acute', 'ultra-fast', 'fast', '3-6mo', '12-24mo'],  # adjust as needed
        ordered=True
    )

    model = smf.mixedlm("FA_diff ~ timepoint + Region", 
                        fa_long_wm_data_roi_567_combi, 
                        groups=fa_long_wm_data_roi_567_combi["patient_id"])
    
    result = model.fit(method='powell')

    # # Output mixed effect model results
    # print("Mixed effects model summary:")
    # print(result.summary())

    # print("\nFixed Effects Parameters:")
    # print(result.fe_params)

    # print("\nRandom Effects Parameters:")
    # print(result.cov_re)

    print_lme_summary_precise(result, precision=6)
    # sys.exit()

    # with posterior as default region: 
    fa_long_wm_data_roi_567_combi_post=fa_long_wm_data_roi_567_combi.copy()
    fa_long_wm_data_roi_567_combi_post["Region"] = pd.Categorical(fa_long_wm_data_roi_567_combi["Region"], categories=["posterior", "anterior"])

    model_posterior = smf.mixedlm("FA_diff ~ timepoint + Region", 
                        fa_long_wm_data_roi_567_combi_post, 
                        groups=fa_long_wm_data_roi_567_combi["patient_id"])
    
    result_post = model_posterior.fit(method='powell')

    # Step 4: Output results
    print("Mixed effects model summary for posterior region as baseline:")
    print_lme_summary_precise(result_post, precision=6)
    # sys.exit()
    # print(result_post.summary())

    # print("\nFixed Effects Parameters:")
    # print(result_post.fe_params)

    # print("\nRandom Effects Parameters:")
    # print(result_post.cov_re)

    ##############################
    ######## PLOTTING LME

    # create_timepoint_boxplot_LME_dti(df=wm_data_roi_567_combi, parameter='fa', result=result, timepoints=['ultra-fast', 'fast', 'acute', '3-6mo', '12-24mo'])
    ##################################

    ###### WHY DO LME? 
    # Check if linear mixed effect (patient as random effect) adds any value
    print("\n=== Formal Test for Random Effects ===")
    # Compare fixed vs mixed effects models using likelihood ratio test
    from scipy import stats

    try:
        # Both models must use ML estimation for valid comparison
        model_fixed = smf.ols("FA_diff ~ timepoint + Region", data=fa_long_wm_data_roi_567_combi)
        result_fixed = model_fixed.fit()
        ll_fixed = result_fixed.llf
        
        model_mixed = smf.mixedlm("FA_diff ~ timepoint + Region",
                                fa_long_wm_data_roi_567_combi,
                                groups=fa_long_wm_data_roi_567_combi["patient_id"])
        result_mixed = model_mixed.fit(method='powell', reml=False)  # Use ML, not REML
        ll_mixed = result_mixed.llf
        
        # Likelihood ratio test
        lr_stat = 2 * (ll_mixed - ll_fixed)
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)  # df=1 for one random effect
        
        print(f"Fixed effects LL: {ll_fixed:.4f}")
        print(f"Mixed effects LL: {ll_mixed:.4f}")
        print(f"LR statistic: {lr_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print("Random effects needed" if p_value < 0.05 else "Fixed effects sufficient")
    except:
        print("Could not perform likelihood ratio test")



    # ###########################
    # ########## FIXED EFFECT WITH CLUSTERED STANDARD ERRORS (accounts for repeated measures by using clustering)
    # Fixed effects model with robust standard errors clustered by patient_id
    # to account for within-patient correlation across timepoints and regions
    # #####################################################
    # # LINEAR FIXED EFFECTS MODEL WITH COMBI DATA AND CLUSTERED SEs
    # # H_0: There is no statistically significant difference between
    # # FA in Control vs. Craniectomy for anterior and posterior ROIs.
    # # i.e. H_0: FA_diff = FA_{control} - FA_{craniectomy} = 0
    # # Fixed effects equation with Timepoint and Region as covariates.
    # # Y_{jk} = \beta_0 + \sum_{t=1}^{T-1} \beta_{1t} \cdot \text{Timepoint}_{jt} + \beta_2 \cdot \text{Region}_k + \varepsilon_{jk}
    # # $Y_{jk}$: FA difference (control - craniectomy) at timepoint $j$, in region $k$
    # # $\beta_0$: Intercept (mean FA difference at reference timepoint and region)
    # # $\beta_{1t}$: Coefficient for timepoint $t$ (excluding the reference level)
    # # $\text{Timepoint}_{jt}$: Indicator variable (1 if observation is at timepoint $t$, else 0)
    # # $\beta_2$: Coefficient for region (e.g., anterior vs posterior)
    # # $\text{Region}_k$: Indicator variable for brain region (e.g., 0 = anterior, 1 = posterior)
    # # $\varepsilon_{jk}$: Residual error, where $\varepsilon_{jk} \sim \mathcal{N}(0, \sigma^2)$
    # # print(wm_data_roi_567_combi.columns)
    # # fa_long_wm_data_roi_567_combi = pd.melt(wm_data_roi_567_combi,
    # # id_vars=['patient_id', 'timepoint'],
    # # value_vars=['fa_anterior_diff', 'fa_posterior_diff'],
    # # var_name='Region',
    # # value_name='FA_diff')
    # # print(fa_long_wm_data_roi_567_combi)
    # # # rename regions to anterior and posterior
    # # fa_long_wm_data_roi_567_combi['Region'] = fa_long_wm_data_roi_567_combi['Region'].map({
    # # 'fa_anterior_diff': 'anterior',
    # # 'fa_posterior_diff': 'posterior'
    # # })
    # # # Order timepoints, so that the first one is reference
    # # # Ensure 'timepoint' is treated as a categorical variable
    # # fa_long_wm_data_roi_567_combi['timepoint'] = pd.Categorical(
    # # fa_long_wm_data_roi_567_combi['timepoint'],
    # # categories=['acute', 'ultra-fast', 'fast', '3-6mo', '12-24mo'], # adjust as needed
    # # ordered=True
    # # )

    # Fixed effects model with clustered standard errors (anterior as reference region)
    import statsmodels.formula.api as smf
    model_fixed = smf.ols("FA_diff ~ timepoint + Region", data=fa_long_wm_data_roi_567_combi)
    result_fixed = model_fixed.fit(cov_type='cluster', cov_kwds={'groups': fa_long_wm_data_roi_567_combi['patient_id']})

    # Output fixed effects model results with clustered SEs
    print("Fixed effects model summary (clustered SEs by patient_id):")
    print(result_fixed.summary())
    print("\nFixed Effects Parameters:")
    print(result_fixed.params)
    print("Fixed effects model summary (clustered SEs by patient_id):")
    # print_lme_summary_precise(result_fixed, precision=6)
    print_fixed_effects_summary_precise(result_fixed, precision=8)
    # Check the actual p-values with high precision
    print("Actual p-values:")
    for param_name in result_fixed.params.index:
        p_val = result_fixed.pvalues[param_name]
        print(f"{param_name}: {p_val:.10e}")  # Scientific notation with 10 decimal places

    

    # Fixed effects model with posterior as reference region
    # fa_long_wm_data_roi_567_combi_post = fa_long_wm_data_roi_567_combi.copy()
    # fa_long_wm_data_roi_567_combi_post["Region"] = pd.Categorical(
    # fa_long_wm_data_roi_567_combi["Region"],
    # categories=["posterior", "anterior"]
    # )
    model_fixed_posterior = smf.ols("FA_diff ~ timepoint + Region", data=fa_long_wm_data_roi_567_combi_post)
    result_fixed_post = model_fixed_posterior.fit(cov_type='cluster', cov_kwds={'groups': fa_long_wm_data_roi_567_combi_post['patient_id']})

    # Output results with posterior as reference and clustered SEs
    print("Fixed effects model summary (posterior reference, clustered SEs by patient_id):")
    # print(result_fixed_post.summary())
    # print("\nFixed Effects Parameters:")
    # print(result_fixed_post.params)
    print_fixed_effects_summary_precise(result_fixed_post, precision=8)
    

    # create_timepoint_boxplot_LME_dti(df=wm_data_roi_567_combi, parameter='fa', result=result, fixed_effects_result=result_fixed)
    # create_timepoint_boxplot_LME_dti(df=wm_data_roi_567_combi, parameter='fa', result=result, fixed_effects_result=result_fixed, fixed_only=True)

    

    ##### PLOT FIXED EFFECT ON BOX PLOT




    ### COMBINE RESULTS wm_data_roi_567 with area data
    area_df=pd.read_csv('Image_Processing_Scripts/area_data.csv')
    batch2_area_df=pd.read_csv('Image_Processing_Scripts/batch2_area_data.csv')
    # add batch2_area_df to area_df
    area_df = pd.concat([area_df, batch2_area_df], ignore_index=True)

    # print(area_df)

    # print(wm_data_roi_567)

    # # Step 1: Compute max area_diff for each patient_id
    # max_area_diff = area_df.groupby('patient_id')['area_diff'].max().reset_index()
    # max_area_diff.rename(columns={'area_diff': 'peak_herniation'}, inplace=True)


    
    # # Step 2: Merge this information into wm_data_roi_567
    # wm_data_roi_567 = wm_data_roi_567.merge(max_area_diff, on='patient_id', how='left')



    # # Optional: Display or inspect the result
    # print(wm_data_roi_567[['patient_id', 'peak_herniation']])

    # Ensure patient_id is of the same type in both dataframes
    area_df['patient_id'] = area_df['patient_id'].astype(str)
    area_df['timepoint'] = area_df['timepoint'].astype(str)

    wm_roi=wm_data_roi_567.copy()
    wm_roi['patient_id'] = wm_roi['patient_id'].astype(str)

    print(area_df['timepoint'])
    print(wm_roi)

    # timepoint_order = ["ultra-fast", "fast", "acute", "3mo", "6mo", "12mo", "24mo"]
    timepoint_order = [
        "ultra-fast", "fast", "acute", "3mo", "6mo", "12mo", "24mo",
        "39", "41", "48", "96", "144", "336", "354", "376", "490", "588",
        "4310", "4311", "4378", "4920", "8659", "8888", "9305", "9672",
        "10079", "18046", "18728", "36840"
    ]


    # Convert timepoint column to categorical type with your custom order
    area_df['timepoint'] = pd.Categorical(area_df['timepoint'], categories=timepoint_order, ordered=True)

    # Sort by patient_id and the ordered timepoint
    area_df = area_df.sort_values(by=['patient_id', 'timepoint']).reset_index()


    # print(area_df)

    # add area_df['area_diff'] column to wm_roi, maintaining this sorted order (same order in both df)
    
    area_df = area_df[~((area_df['patient_id'] == '20942') & (area_df['timepoint'] == '24mo'))]
    # Remove patient_id 9GfT823 with timepoint 39 from area_df
    area_df = area_df[~((area_df['patient_id'] == '9GfT823') & (area_df['timepoint'] == '39'))]

    # Reset index
    area_df = area_df.reset_index(drop=True)
    # Remove the index column from area_df if it exists
    if 'index' in area_df.columns:
        area_df = area_df.drop('index', axis=1)
    # Remove patient 20174 from area_df
    area_df_filtered = area_df[area_df['patient_id'] != '20174'].reset_index(drop=True)
    # print(area_df_filtered)
    # print(wm_roi)
    


    wm_roi['area_diff'] = area_df_filtered['area_diff'].values
    # print(wm_roi)

    wm_fa_hern=wm_roi.copy()
    wm_fa_hern=wm_fa_hern[['patient_id', 'timepoint', 'fa_anterior_diff', 'fa_posterior_diff', 'area_diff']]
    # print(wm_fa_hern)

    wm_fa_hern_combi=wm_fa_hern.copy()
    wm_fa_hern_combi['timepoint']=wm_fa_hern['timepoint'].replace({
        '3mo' : '3-6mo',
        '6mo' : '3-6mo',
        '12mo' : '12-24mo',
        '24mo' : '12-24mo'
    })
    wm_fa_hern_combi = wm_fa_hern_combi.drop_duplicates(subset=['patient_id', 'timepoint'], keep='first')

    # print(wm_fa_hern_combi)
    # wm_fa_hern_combi_matrix=data_availability_matrix(
    #     data=wm_fa_hern_combi, 
    #     timepoints=["ultra-fast", "fast", "acute", "3-6mo", "12-24mo"],
    #     diff_column='fa_anterior_diff',
    #     filename='data_availability_combi_area_diff_dti.png')

    # Redo area diff model with FA_diff as covariate

    # Model 1: FA only
    model1 = smf.mixedlm("area_diff ~ fa_anterior_diff + fa_posterior_diff", 
                        data=wm_fa_hern_combi, 
                        groups=wm_fa_hern_combi['patient_id'])
    result1 = model1.fit()

    # Model 2: FA + timepoint
    wm_fa_hern_combi['timepoint']=pd.Categorical(
        wm_fa_hern_combi['timepoint'],
        categories=['acute', 'ultra-fast', 'fast', '3-6mo', '12-24mo'], 
        ordered=True
    )
    model2 = smf.mixedlm("area_diff ~ timepoint + fa_anterior_diff + fa_posterior_diff", 
                        data=wm_fa_hern_combi, 
                        groups=wm_fa_hern_combi['patient_id'])
    result2 = model2.fit()

    # Compare: Are FA effects consistent across both models?
    print("\nLME with no timepoint:")
    print(result1.summary())
    print(result1.params)
    print("\nLME summary with timepoint:")
    print(result2.summary())
    print(result2.params)



    
    # fig1, axes1 = create_fa_area_correlation_plot(wm_fa_hern_combi, show_timepoints=True)
    # fig2, axes2 = create_fa_area_correlation_plot(wm_fa_hern_combi, show_timepoints=False)

    # # Model results for prediction plot
    # model_results = {
    #     'intercept': 217.92,
    #     'fa_anterior_coef': 4747.88
    # }
    # fig3, ax3 = create_fa_area_model_visualization(wm_fa_hern_combi, model_results)


    ####################
    # DOES HERNIATION CAUSE FA_DIFF? 
    # Model 4: Area predicts FA anterior (primary)
    print("****************************\nNEW_MODELS")
    model4 = smf.mixedlm("fa_anterior_diff ~ area_diff", 
                        data=wm_fa_hern_combi, 
                        groups=wm_fa_hern_combi['patient_id'])

    # Model 5: Area predicts FA posterior (comparison) 
    model5 = smf.mixedlm("fa_posterior_diff ~ area_diff", 
                        data=wm_fa_hern_combi, 
                        groups=wm_fa_hern_combi['patient_id'])

    # Model 6: Area predicts FA anterior (with timepoint control)
    model6 = smf.mixedlm("fa_anterior_diff ~ area_diff + timepoint", 
                        data=wm_fa_hern_combi, 
                        groups=wm_fa_hern_combi['patient_id'])
    

    result4=model4.fit()
    result5=model5.fit()
    result6=model6.fit()

    print("\nArea Predicts FA Anterior LME Summary:")
    print_lme_summary_precise(result4, precision=6)

    
    print("\nArea Predicts FA Posterior LME Summary:")
    print_lme_summary_precise(result5, precision=6)

    print("\nArea Predicts FA Anterior with Timepoint Control LME Summary:")
    print_lme_summary_precise(result6, precision=6)


    # sys.exit()





    # # Extract the actual random intercept estimates (û_i values)
    # random_intercepts = result4.random_effects
    # print("Random intercept estimates:")
    # for patient_id, effects in random_intercepts.items():
    #     print(f"Patient {patient_id}: {effects['Group']:.6f}")

    # # Calculate variance manually
    # intercept_values = [effects['Group'] for effects in random_intercepts.values()]
    # manual_var = np.var(intercept_values, ddof=1)
    # print(f"\nManual calculation of group variance: {manual_var:.6f}")
    # print(f"Model estimate: {result4.cov_re}")
    # print("\nArea Predicts FA Posterior LME Summary:")
    # print(result5.summary())
    # print(result5.params)
    # print("\nArea Predicts FA Anterior with Timepoint Control LME Summary:")
    # print(result6.summary())
    # print(result6.params)

    # Usage example:
    # create_area_predicts_fa_plot(wm_fa_hern_combi, result4, result5, show_combined=True)
    # create_area_predicts_fa_plot(wm_fa_hern_combi, result4, result5, show_combined=False)

    # sys.exit()

    ##########################
    ##########################
    ###########################
    ##########################
    ##########################
    ###########################
    ##########################
    ##########################
    ###########################
    ##########################
    ##########################
    ###########################
    # REPEAT FOR MD. 

    md_long_wm_data_roi_567_combi = pd.melt(wm_data_roi_567_combi,
                  id_vars=['patient_id', 'timepoint'],
                  value_vars=['md_anterior_diff', 'md_posterior_diff'],
                  var_name='Region',
                  value_name='MD_diff')
    
    print(md_long_wm_data_roi_567_combi)

    print("MD data info:")
    print(f"md_anterior_diff null count: {wm_data_roi_567_combi['md_anterior_diff'].isnull().sum()}")
    print(f"md_posterior_diff null count: {wm_data_roi_567_combi['md_posterior_diff'].isnull().sum()}")

    print("\nAfter melting:")
    print(f"md_long_wm_data_roi_567_combi shape: {md_long_wm_data_roi_567_combi.shape}")
    print(f"Null values in MD_diff: {md_long_wm_data_roi_567_combi['MD_diff'].isnull().sum()}")

    # Keep record of rows with missing MD_diff values for later filtering
    md_missing_mask = md_long_wm_data_roi_567_combi['MD_diff'].isna()
    md_missing_rows = md_long_wm_data_roi_567_combi[md_missing_mask][['patient_id', 'timepoint', 'Region']].copy()
    print(f"Rows with missing MD data that will be removed:")
    print(md_missing_rows)
    print(f"Total missing rows: {len(md_missing_rows)}")
    
    # Remove rows with missing MD_diff values
    md_long_wm_data_roi_567_combi = md_long_wm_data_roi_567_combi.dropna(subset=['MD_diff'])
    print(f"After removing NaNs: {md_long_wm_data_roi_567_combi.shape}")

    # sys.exit()

    # rename regions to anterior and posterior
    md_long_wm_data_roi_567_combi['Region'] = md_long_wm_data_roi_567_combi['Region'].map({
        'md_anterior_diff': 'anterior',
        'md_posterior_diff': 'posterior'
    })

    # Order timepoints, so that the first one is reference
    # Ensure 'timepoint' is treated as a categorical variable
    md_long_wm_data_roi_567_combi['timepoint'] = pd.Categorical(
        md_long_wm_data_roi_567_combi['timepoint'],
        categories=['acute', 'ultra-fast', 'fast', '3-6mo', '12-24mo'],  # adjust as needed
        ordered=True
    )

    model = smf.mixedlm("MD_diff ~ timepoint + Region", 
                        md_long_wm_data_roi_567_combi, 
                        groups=md_long_wm_data_roi_567_combi["patient_id"])
    
    result = model.fit(method='powell')

    # Output mixed effect model results
    print("Mixed effects model summary:")
    
    # print(result.summary())
    print_lme_summary_precise(result,precision=6)
    # sys.exit()

    # print("\nFixed Effects Parameters:")
    # print(result.fe_params)

    # print("\nRandom Effects Parameters:")
    # print(result.cov_re)

    # with posterior as default region: 
    md_long_wm_data_roi_567_combi_post=md_long_wm_data_roi_567_combi.copy()
    md_long_wm_data_roi_567_combi_post["Region"] = pd.Categorical(md_long_wm_data_roi_567_combi["Region"], categories=["posterior", "anterior"])

    model_posterior = smf.mixedlm("MD_diff ~ timepoint + Region", 
                        md_long_wm_data_roi_567_combi_post, 
                        groups=md_long_wm_data_roi_567_combi["patient_id"])
    
    result_post = model_posterior.fit(method='powell')

    # Step 4: Output results
    print("Mixed effects model summary (posterior as ref):")
    # print(result_post.summary())
    print_lme_summary_precise(result_post,precision=6)
    # sys.exit()

    # print("\nFixed Effects Parameters:")
    # print(result_post.fe_params)

    # print("\nRandom Effects Parameters:")
    # print(result_post.cov_re)

    ##############################
    ######## PLOTTING LME

    # create_timepoint_boxplot_LME_dti(df=wm_data_roi_567_combi, parameter='md', result=result, timepoints=['ultra-fast', 'fast', 'acute', '3-6mo', '12-24mo'])
    ##################################

    ###### WHY DO LME? 
    # Check if linear mixed effect (patient as random effect) adds any value
    print("\n=== Formal Test for Random Effects ===")
    # Compare fixed vs mixed effects models using likelihood ratio test
    from scipy import stats

    try:
        # Both models must use ML estimation for valid comparison
        model_fixed = smf.ols("MD_diff ~ timepoint + Region", data=md_long_wm_data_roi_567_combi)
        result_fixed = model_fixed.fit()
        ll_fixed = result_fixed.llf
        
        model_mixed = smf.mixedlm("MD_diff ~ timepoint + Region",
                                md_long_wm_data_roi_567_combi,
                                groups=md_long_wm_data_roi_567_combi["patient_id"])
        result_mixed = model_mixed.fit(method='powell', reml=False)  # Use ML, not REML
        ll_mixed = result_mixed.llf
        
        # Likelihood ratio test
        lr_stat = 2 * (ll_mixed - ll_fixed)
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)  # df=1 for one random effect
        
        print(f"Fixed effects LL: {ll_fixed:.4f}")
        print(f"Mixed effects LL: {ll_mixed:.4f}")
        print(f"LR statistic: {lr_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print("Random effects needed" if p_value < 0.05 else "Fixed effects sufficient")
    except:
        print("Could not perform likelihood ratio test")


    # ###########################
    # ########## FIXED EFFECT WITH CLUSTERED STANDARD ERRORS (accounts for repeated measures by using clustering)
    # Fixed effects model with robust standard errors clustered by patient_id
    # to account for within-patient correlation across timepoints and regions
    # #####################################################
    # # LINEAR FIXED EFFECTS MODEL WITH COMBI DATA AND CLUSTERED SEs
    # # H_0: There is no statistically significant difference between
    # # MD in Control vs. Craniectomy for anterior and posterior ROIs.
    # # i.e. H_0: MD_diff = MD_{control} - MD_{craniectomy} = 0
    # # Fixed effects equation with Timepoint and Region as covariates.
    # # Y_{jk} = \beta_0 + \sum_{t=1}^{T-1} \beta_{1t} \cdot \text{Timepoint}_{jt} + \beta_2 \cdot \text{Region}_k + \varepsilon_{jk}
    # # $Y_{jk}$: MD difference (control - craniectomy) at timepoint $j$, in region $k$
    # # $\beta_0$: Intercept (mean MD difference at reference timepoint and region)
    # # $\beta_{1t}$: Coefficient for timepoint $t$ (excluding the reference level)
    # # $\text{Timepoint}_{jt}$: Indicator variable (1 if observation is at timepoint $t$, else 0)
    # # $\beta_2$: Coefficient for region (e.g., anterior vs posterior)
    # # $\text{Region}_k$: Indicator variable for brain region (e.g., 0 = anterior, 1 = posterior)
    # # $\varepsilon_{jk}$: Residual error, where $\varepsilon_{jk} \sim \mathcal{N}(0, \sigma^2)$
    # # print(wm_data_roi_567_combi.columns)
    # # md_long_wm_data_roi_567_combi = pd.melt(wm_data_roi_567_combi,
    # # id_vars=['patient_id', 'timepoint'],
    # # value_vars=['md_anterior_diff', 'md_posterior_diff'],
    # # var_name='Region',
    # # value_name='MD_diff')
    # # print(md_long_wm_data_roi_567_combi)
    # # # rename regions to anterior and posterior
    # # md_long_wm_data_roi_567_combi['Region'] = md_long_wm_data_roi_567_combi['Region'].map({
    # # 'md_anterior_diff': 'anterior',
    # # 'md_posterior_diff': 'posterior'
    # # })
    # # # Order timepoints, so that the first one is reference
    # # # Ensure 'timepoint' is treated as a categorical variable
    # # md_long_wm_data_roi_567_combi['timepoint'] = pd.Categorical(
    # # md_long_wm_data_roi_567_combi['timepoint'],
    # # categories=['acute', 'ultra-fast', 'fast', '3-6mo', '12-24mo'], # adjust as needed
    # # ordered=True
    # # )

    # Fixed effects model with clustered standard errors (anterior as reference region)
    import statsmodels.formula.api as smf
    model_fixed = smf.ols("MD_diff ~ timepoint + Region", data=md_long_wm_data_roi_567_combi)
    result_fixed = model_fixed.fit(cov_type='cluster', cov_kwds={'groups': md_long_wm_data_roi_567_combi['patient_id']})

    # Output fixed effects model results with clustered SEs
    print("Fixed effects model summary (clustered SEs by patient_id):")
    # print(result_fixed.summary())
    # print("\nFixed Effects Parameters:")
    # print(result_fixed.params)
    print_fixed_effects_summary_precise(result_fixed, precision=6)
    # sys.exit()

    # Fixed effects model with posterior as reference region
    # md_long_wm_data_roi_567_combi_post = md_long_wm_data_roi_567_combi.copy()
    # md_long_wm_data_roi_567_combi_post["Region"] = pd.Categorical(
    # md_long_wm_data_roi_567_combi["Region"],
    # categories=["posterior", "anterior"]
    # )
    model_fixed_posterior = smf.ols("MD_diff ~ timepoint + Region", data=md_long_wm_data_roi_567_combi_post)
    result_fixed_post = model_fixed_posterior.fit(cov_type='cluster', cov_kwds={'groups': md_long_wm_data_roi_567_combi_post['patient_id']})

    # Output results with posterior as reference and clustered SEs
    print("Fixed effects model summary (posterior reference, clustered SEs by patient_id):")
    # print(result_fixed_post.summary())
    # print("\nFixed Effects Parameters:")
    # print(result_fixed_post.params)
    print_fixed_effects_summary_precise(result_fixed_post,precision=6)
    # sys.exit()

    # create_timepoint_boxplot_LME_dti(df=wm_data_roi_567_combi, parameter='md', result=result, fixed_effects_result=result_fixed)
    # create_timepoint_boxplot_LME_dti(df=wm_data_roi_567_combi, parameter='md', result=result, fixed_effects_result=result_fixed, fixed_only=True)
    ##### PLOT FIXED EFFECT ON BOX PLOT

    ### COMBINE RESULTS wm_data_roi_567 with area data
    area_df=pd.read_csv('Image_Processing_Scripts/area_data.csv')
    batch2_area_df=pd.read_csv('Image_Processing_Scripts/batch2_area_data.csv')
    # add batch2_area_df to area_df
    area_df = pd.concat([area_df, batch2_area_df], ignore_index=True)

    # print(area_df)

    # print(wm_data_roi_567)

    # # Step 1: Compute max area_diff for each patient_id
    # max_area_diff = area_df.groupby('patient_id')['area_diff'].max().reset_index()
    # max_area_diff.rename(columns={'area_diff': 'peak_herniation'}, inplace=True)


    
    # # Step 2: Merge this information into wm_data_roi_567
    # wm_data_roi_567 = wm_data_roi_567.merge(max_area_diff, on='patient_id', how='left')



    # # Optional: Display or inspect the result
    # print(wm_data_roi_567[['patient_id', 'peak_herniation']])

    # Ensure patient_id is of the same type in both dataframes
    area_df['patient_id'] = area_df['patient_id'].astype(str)
    area_df['timepoint'] = area_df['timepoint'].astype(str)

    wm_roi=wm_data_roi_567.copy()
    wm_roi['patient_id'] = wm_roi['patient_id'].astype(str)

    print(area_df['timepoint'])
    print(wm_roi)

    # timepoint_order = ["ultra-fast", "fast", "acute", "3mo", "6mo", "12mo", "24mo"]
    timepoint_order = [
        "ultra-fast", "fast", "acute", "3mo", "6mo", "12mo", "24mo",
        "39", "41", "48", "96", "144", "336", "354", "376", "490", "588",
        "4310", "4311", "4378", "4920", "8659", "8888", "9305", "9672",
        "10079", "18046", "18728", "36840"
    ]


    # Convert timepoint column to categorical type with your custom order
    area_df['timepoint'] = pd.Categorical(area_df['timepoint'], categories=timepoint_order, ordered=True)

    # Sort by patient_id and the ordered timepoint
    area_df = area_df.sort_values(by=['patient_id', 'timepoint']).reset_index()


    # print(area_df)

    # add area_df['area_diff'] column to wm_roi, maintaining this sorted order (same order in both df)
    
    area_df = area_df[~((area_df['patient_id'] == '20942') & (area_df['timepoint'] == '24mo'))]
    # Remove patient_id 9GfT823 with timepoint 39 from area_df
    area_df = area_df[~((area_df['patient_id'] == '9GfT823') & (area_df['timepoint'] == '39'))]

    # Reset index
    area_df = area_df.reset_index(drop=True)
    # Remove the index column from area_df if it exists
    if 'index' in area_df.columns:
        area_df = area_df.drop('index', axis=1)

    # Identify which rows in wm_roi have missing MD data that need to be removed
    md_rows_to_remove = []
    for _, missing_row in md_missing_rows.iterrows():
        mask = None  # Initialize mask
        if missing_row['Region'] == 'anterior':
            mask = (wm_roi['patient_id'] == missing_row['patient_id']) & \
                   (wm_roi['timepoint'] == missing_row['timepoint']) & \
                   (wm_roi['md_anterior_diff'].isna())
        elif missing_row['Region'] == 'posterior':
            mask = (wm_roi['patient_id'] == missing_row['patient_id']) & \
                   (wm_roi['timepoint'] == missing_row['timepoint']) & \
                   (wm_roi['md_posterior_diff'].isna())
        
        if mask is not None:
            md_rows_to_remove.extend(wm_roi[mask].index.tolist())

    # Remove patient 20174 from area_df
    area_df_filtered = area_df[area_df['patient_id'] != '20174'].reset_index(drop=True)
    
    # Remove the same rows from both dataframes that have missing MD data
    if md_rows_to_remove:
        area_df_filtered = area_df_filtered.drop(md_rows_to_remove, errors='ignore').reset_index(drop=True)
        wm_roi = wm_roi.drop(md_rows_to_remove).reset_index(drop=True)

    wm_roi['area_diff'] = area_df_filtered['area_diff'].values
    # print(wm_roi)

    wm_md_hern=wm_roi.copy()
    wm_md_hern=wm_md_hern[['patient_id', 'timepoint', 'md_anterior_diff', 'md_posterior_diff', 'area_diff']]
    
    print(f"wm_md_hern shape: {wm_md_hern.shape}")
    # print(wm_md_hern)

    wm_md_hern_combi=wm_md_hern.copy()
    wm_md_hern_combi['timepoint']=wm_md_hern['timepoint'].replace({
        '3mo' : '3-6mo',
        '6mo' : '3-6mo',
        '12mo' : '12-24mo',
        '24mo' : '12-24mo'
    })
    wm_md_hern_combi = wm_md_hern_combi.drop_duplicates(subset=['patient_id', 'timepoint'], keep='first')
    
    # Additional safety check - remove any remaining NaNs
    wm_md_hern_combi = wm_md_hern_combi.dropna(subset=['md_anterior_diff', 'md_posterior_diff', 'area_diff'])
    
    print(f"wm_md_hern_combi final shape: {wm_md_hern_combi.shape}")
    print(wm_md_hern_combi)
    # sys.exit()
    # wm_md_hern_combi_matrix=data_availability_matrix(
    #     data=wm_md_hern_combi, 
    #     timepoints=["ultra-fast", "fast", "acute", "3-6mo", "12-24mo"],
    #     diff_column='md_anterior_diff',
    #     filename='data_availability_combi_area_diff_dti.png')

    # Redo area diff model with MD_diff as covariate

    # Model 1: MD only
    model1 = smf.mixedlm("area_diff ~ md_anterior_diff + md_posterior_diff", 
                        data=wm_md_hern_combi, 
                        groups=wm_md_hern_combi['patient_id'])
    result1 = model1.fit()

    # Model 2: MD + timepoint
    wm_md_hern_combi['timepoint']=pd.Categorical(
        wm_md_hern_combi['timepoint'],
        categories=['acute', 'ultra-fast', 'fast', '3-6mo', '12-24mo'], 
        ordered=True
    )
    model2 = smf.mixedlm("area_diff ~ timepoint + md_anterior_diff + md_posterior_diff", 
                        data=wm_md_hern_combi, 
                        groups=wm_md_hern_combi['patient_id'])
    result2 = model2.fit()

    # Compare: Are MD effects consistent across both models?
    print("\nLME with no timepoint:")
    # print(result1.summary())
    # print(result1.params)
    print_lme_summary_precise(result1,precision=6)
    # sys.exit()

    print("\nLME summary with timepoint:")
    # print(result2.summary())
    # print(result2.params)
    print_lme_summary_precise(result2,precision=6)
    # sys.exit()

    ####################
    # DOES HERNIATION CAUSE MD_DIFF? 
    # Model 4: Area predicts MD anterior (primary)
    print("****************************\nNEW_MODELS")
    model4 = smf.mixedlm("md_anterior_diff ~ area_diff", 
                        data=wm_md_hern_combi, 
                        groups=wm_md_hern_combi['patient_id'])

    # Model 5: Area predicts MD posterior (comparison) 
    model5 = smf.mixedlm("md_posterior_diff ~ area_diff", 
                        data=wm_md_hern_combi, 
                        groups=wm_md_hern_combi['patient_id'])

    # Model 6: Area predicts MD anterior (with timepoint control)
    model6 = smf.mixedlm("md_anterior_diff ~ area_diff + timepoint", 
                        data=wm_md_hern_combi, 
                        groups=wm_md_hern_combi['patient_id'])

    # Model 7: Area predicts MD posterior (with timepoint control)
    model7 = smf.mixedlm("md_posterior_diff ~ area_diff + timepoint", 
                        data=wm_md_hern_combi, 
                        groups=wm_md_hern_combi['patient_id'])
    

    result4=model4.fit()
    result5=model5.fit()
    result6=model6.fit()
    result7=model7.fit()

    print("\nArea Predicts MD Anterior LME Summary:")
    print_lme_summary_precise(result4,precision=8)
    # sys.exit()
    print("\nArea Predicts MD Posterior LME Summary:")
    # print(result5.summary())
    # print(result5.params)
    print_lme_summary_precise(result5,precision=8)
    # sys.exit()

    print("\nArea Predicts MD Anterior with Timepoint Control LME Summary:")
    print_lme_summary_precise(result6,precision=8)
    # sys.exit()
    print("\nArea Predicts MD Posterior with Timepoint Control LME Summary:")
    # print(result7.summary())
    # print(result7.params)
    print_lme_summary_precise(result7,precision=8)
    sys.exit()


    # Usage example:
    create_area_predicts_md_plot(wm_md_hern_combi, result4, result5, show_combined=True)
    create_area_predicts_md_plot(wm_md_hern_combi, result4, result5, show_combined=False)






    

    sys.exit()










































    # Compute max area_diff per patient
    max_area_diff = area_df.groupby('patient_id')['area_diff'].max().reset_index()

    # Optional: Rename column for clarity
    max_area_diff.rename(columns={'area_diff': 'peak_herniation'}, inplace=True)

    # Merge into wm_data_roi_567
    wm_roi = wm_roi.merge(max_area_diff, on='patient_id', how='left')

    # Create a new column `peak_fa_diff` as the max of the two columns per row
    wm_roi['peak_fa_diff'] = wm_roi[['fa_anterior_diff', 'fa_posterior_diff']].max(axis=1)

    # Put patient_id, max_FA_diff and peak_herniation into new data frame (no timepoint data)
    # Step 1: Group by patient and get max FA diff
    max_fa_diff = wm_roi.groupby('patient_id')['peak_fa_diff'].max().reset_index()
    max_fa_diff.rename(columns={'peak_fa_diff': 'max_FA_diff'}, inplace=True)

    # Step 2: Get peak herniation (you've already done this, but just in case)
    peak_herniation = wm_roi[['patient_id', 'peak_herniation']].drop_duplicates()

    # Step 3: Merge both into one dataframe
    summary_df = max_fa_diff.merge(peak_herniation, on='patient_id', how='left')



    



    print(summary_df)

    #### PLOTTING
       
    # Apply the publication style
    set_publication_style()

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x=summary_df['peak_herniation'], y=summary_df['max_FA_diff'], color='blue')
    # Labels and title
    plt.xlabel('Peak Herniation')
    plt.xlim(-500,1750)
    plt.ylim(0,0.2)
    plt.ylabel('Max FA Difference')
    plt.title('Relationship between FA Difference and Peak Herniation')

    # Show plot
    plt.tight_layout()
    plt.savefig('DTI_Processing_Scripts/fa_diff_vs_peak_herniation.png', dpi=300)
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter6/Figs/fa_diff_vs_peak_herniation.png', dpi=600)
    plt.close()



    















    print("\n\nScript complete!")


