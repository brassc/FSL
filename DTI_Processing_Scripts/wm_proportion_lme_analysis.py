#!/usr/bin/env python3
"""
WM Proportion Longitudinal Statistical Analysis
Tests the null hypothesis: WM proportion does not change significantly over time
Disaggregated by region: anterior, posterior, baseline_anterior, baseline_posterior
Uses Linear Mixed Effects models to account for repeated measures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import set_publication_style
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Image_Processing_Scripts.set_publication_style import set_publication_style

def map_timepoint_to_string(numeric_timepoint):
    """Convert numeric timepoint to string timepoint."""
    timepoints = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
    timepoint_ranges = [
        (0, 48),        # ultra-fast: 0-48 hours (0-2 days)
        (48, 192),      # fast: 48-192 hours (2-8 days)
        (192, 1008),    # acute: 192-1008 hours (8-42 days)
        (1008, 4296),   # 3mo: 42-179 days
        (4296, 6672),   # 6mo: 179-278 days
        (6672, 12960),  # 12mo: 278-540 days
        (12960, 500000) # 24mo: 540+ days
    ]

    for i, (min_val, max_val) in enumerate(timepoint_ranges):
        if min_val <= numeric_timepoint < max_val:
            return timepoints[i]

    return timepoints[-1]

def process_timepoint_data(df):
    """Process and standardize timepoint values."""
    df['patient_id'] = df['patient_id'].astype(str)
    timepoints = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']

    # Check if timepoint is already a string
    string_mask = df['timepoint'].isin(timepoints)
    numeric_mask = ~string_mask & df['timepoint'].apply(lambda x: pd.notnull(pd.to_numeric(x, errors='coerce')))

    # Convert numeric values to their appropriate string representations
    for idx in df[numeric_mask].index:
        try:
            numeric_value = float(df.loc[idx, 'timepoint'])
            df.loc[idx, 'timepoint'] = map_timepoint_to_string(numeric_value)
        except (ValueError, TypeError):
            continue

    return df

def fit_lme_model(df, dv_column, output_file=None):
    """
    Fit LME model: DV ~ timepoint, random intercept for patient_id

    Args:
        df: DataFrame with columns [patient_id, timepoint, dv_column]
        dv_column: Name of dependent variable column
        output_file: Optional file to write model output

    Returns:
        Dictionary with model results
    """
    # Remove NA values
    df_clean = df[['patient_id', 'timepoint', dv_column]].dropna()

    if len(df_clean) < 10:
        print(f"Warning: Only {len(df_clean)} observations for {dv_column}, skipping analysis")
        return None

    # Set timepoint as categorical with 'acute' as reference
    timepoint_order = ['acute', 'ultra-fast', 'fast', '3mo', '6mo', '12mo', '24mo']
    df_clean['timepoint'] = pd.Categorical(
        df_clean['timepoint'],
        categories=timepoint_order,
        ordered=True
    )

    # Fit LME model using statsmodels
    formula = f'{dv_column} ~ timepoint'

    try:
        #         $$Y_{ij} = \beta_0 + \sum_{t=1}^{T-1} \beta_{1t} \cdot
        # \text{Timepoint}{jt} + u_i + \varepsilon{ij}$$

        # Where:
        # - $Y_{ij}$: WM proportion for patient $i$ at observation $j$
        # - $\beta_0$: Intercept (mean WM proportion at acute timepoint,
        # the reference)
        # - $\beta_{1t}$: Fixed effect coefficient for timepoint $t$
        # relative to acute
        # - $\text{Timepoint}_{jt}$: Indicator variable (1 if observation
        # $j$ is at timepoint $t$, 0 otherwise)
        # - $T = 7$: Total number of timepoints (acute, ultra-fast, fast,
        # 3mo, 6mo, 12mo, 24mo)
        # - $t \in 1, ..., 6$: Non-reference timepoints (ultra-fast,
        # fast, 3mo, 6mo, 12mo, 24mo)
        # - $u_i \sim \mathcal{N}(0, \sigma_u^2)$: Random intercept for
        # patient $i$
        # - $\varepsilon_{ij} \sim \mathcal{N}(0, \sigma^2)$: Residual
        # error

        # Fit model
        model = smf.mixedlm(formula, df_clean, groups=df_clean["patient_id"])
        result = model.fit(method='powell')

        # Extract p-values for timepoint effects
        # Check if ANY timepoint coefficient is significant (p < 0.05)
        timepoint_pvals = {}
        for param in result.pvalues.index:
            if param.startswith('timepoint[T.'):
                timepoint_pvals[param] = result.pvalues[param]

        # Check if any timepoint is significant
        min_pval = min(timepoint_pvals.values()) if timepoint_pvals else np.nan
        any_significant = any(p < 0.05 for p in timepoint_pvals.values()) if timepoint_pvals else False

        # Write detailed output if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"Linear Mixed Effects Model Results\n")
                f.write(f"Formula: {formula}\n")
                f.write(f"Groups: patient_id\n")
                f.write(f"=" * 80 + "\n\n")
                f.write(result.summary().as_text())
                f.write(f"\n\n")
                f.write(f"Timepoint coefficients (vs reference 'acute'):\n")
                for param in result.params.index:
                    if param.startswith('timepoint[T.'):
                        f.write(f"  {param}: coef={result.params[param]:.6f}, "
                               f"p={result.pvalues[param]:.6f}")
                        if result.pvalues[param] < 0.05:
                            f.write(" *")
                        f.write("\n")
                f.write(f"\n")
                f.write(f"Minimum p-value across timepoints: {min_pval:.6f}\n")
                f.write(f"Any timepoint significantly different from reference: {any_significant}\n")

        results = {
            'timepoint_pvals': timepoint_pvals,
            'min_pval': min_pval,
            'any_significant': any_significant,
            'n_obs': len(df_clean),
            'n_patients': df_clean['patient_id'].nunique(),
            'result': result
        }

        return results

    except Exception as e:
        print(f"Error fitting LME model for {dv_column}: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_region(df, region, rings=[5, 6, 7], output_dir='DTI_Processing_Scripts/wm_proportion_lme_results'):
    """
    Analyze WM proportion changes over time for a specific region.

    Args:
        df: DataFrame with WM proportion data
        region: Region name (ant, post, baseline_ant, baseline_post)
        rings: List of rings to analyze
        output_dir: Directory to save results
    """
    print(f"\n{'='*80}")
    print(f"Analyzing region: {region.upper()}")
    print(f"{'='*80}")

    # Calculate mean WM proportion across specified rings
    prop_cols = [f'WM_prop_{region}_ring_{ring}' for ring in rings]

    # Check if all columns exist
    missing_cols = [col for col in prop_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns {missing_cols}")
        return None

    # Calculate mean
    df[f'WM_prop_{region}_mean'] = df[prop_cols].mean(axis=1)

    # Prepare data for LME
    lme_data = df[['patient_id', 'timepoint', f'WM_prop_{region}_mean']].copy()
    lme_data = lme_data.dropna()

    print(f"N observations: {len(lme_data)}")
    print(f"N patients: {lme_data['patient_id'].nunique()}")
    print(f"Timepoints: {sorted(lme_data['timepoint'].unique())}")

    # Summary statistics by timepoint
    print(f"\nDescriptive Statistics:")
    summary = lme_data.groupby('timepoint')[f'WM_prop_{region}_mean'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(summary.to_string())

    # Fit LME model
    output_file = os.path.join(output_dir, f'lme_{region}_rings_{min(rings)}-{max(rings)}.txt')
    results = fit_lme_model(lme_data, f'WM_prop_{region}_mean', output_file)

    if results:
        print(f"\n*** LME MODEL RESULTS ***")
        print(f"Minimum p-value across timepoints: {results['min_pval']:.6f}")

        # Print individual timepoint p-values
        print(f"\nIndividual timepoint comparisons:")
        for param, pval in results['timepoint_pvals'].items():
            sig_marker = " *" if pval < 0.05 else ""
            print(f"  {param}: p = {pval:.6f}{sig_marker}")

        if results['any_significant']:
            print(f"\n→ SIGNIFICANT: At least one timepoint differs significantly (p < 0.05)")
        else:
            print(f"\n→ NOT SIGNIFICANT: No timepoints differ significantly (all p ≥ 0.05)")

        print(f"\nDetailed results saved to: {output_file}")

    return results

def create_summary_table(results_dict, output_dir):
    """Create summary table of all LME results."""
    summary_data = []

    for region, results in results_dict.items():
        if results:
            summary_data.append({
                'Region': region,
                'N_observations': results['n_obs'],
                'N_patients': results['n_patients'],
                'Min_p_value': results['min_pval'],
                'Any_significant': 'Yes' if results['any_significant'] else 'No'
            })

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    output_file = os.path.join(output_dir, 'wm_proportion_lme_summary.csv')
    summary_df.to_csv(output_file, index=False)
    print(f"\nSummary table saved to: {output_file}")

    return summary_df

def plot_trajectories_by_region(df, region, rings=[5, 6, 7], output_dir='DTI_Processing_Scripts/wm_proportion_lme_results'):
    """Create trajectory plot for a specific region."""
    # Set publication style for consistent formatting
    set_publication_style()

    # Calculate mean WM proportion
    prop_cols = [f'WM_prop_{region}_ring_{ring}' for ring in rings]
    df[f'WM_prop_{region}_mean'] = df[prop_cols].mean(axis=1)

    # Define timepoint order
    timepoint_order = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']

    # Create categorical timepoint
    df['timepoint_cat'] = pd.Categorical(
        df['timepoint'],
        categories=timepoint_order,
        ordered=True
    )

    # Calculate mean and SEM by timepoint
    summary = df.groupby('timepoint_cat')[f'WM_prop_{region}_mean'].agg(['mean', 'sem', 'count']).reset_index()
    summary = summary.dropna()

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = range(len(summary))
    ax.errorbar(x_pos, summary['mean'], yerr=summary['sem'],
                marker='o', markersize=8, linewidth=2, capsize=5,
                label='Mean ± SEM')

    # Add individual patient trajectories (faint)
    for patient_id in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == patient_id].sort_values('timepoint_cat')
        if len(patient_data) >= 2:
            tp_indices = [timepoint_order.index(tp) for tp in patient_data['timepoint'] if tp in timepoint_order]
            wm_vals = patient_data[f'WM_prop_{region}_mean'].values[:len(tp_indices)]
            ax.plot(tp_indices, wm_vals, alpha=0.15, linewidth=1, color='gray')

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(summary['timepoint_cat'], rotation=45, ha='right')
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('WM Proportion (Mean Rings 5-7)')

    region_title = region.replace('_', ' ').title()
    ax.set_title(f'WM Proportion Over Time: {region_title}')

    ax.legend()

    # Save to results directory (PNG)
    output_file = os.path.join(output_dir, f'wm_proportion_trajectory_{region}.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Trajectory plot saved to: {output_file}")

    # Save to thesis directory (PDF, high resolution)
    thesis_dir = '../Thesis/phd-thesis-template-2.4/Chapter6/Figs'
    if os.path.exists(thesis_dir):
        thesis_file = os.path.join(thesis_dir, f'wm_proportion_trajectory_{region}.pdf')
        plt.savefig(thesis_file, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Trajectory plot saved to thesis: {thesis_file}")

    plt.close()

def plot_trajectories_all(df, rings=[5, 6, 7], output_dir='DTI_Processing_Scripts/wm_proportion_lme_results'):
    """
    Create single combined trajectory plot with all four regions:
    - Anterior craniectomy: blue solid line
    - Posterior craniectomy: red solid line
    - Anterior control (baseline): blue dashed line
    - Posterior control (baseline): red dashed line
    """
    # Set publication style
    set_publication_style()

    # Match jt-test plot font sizes
    plt.rcParams.update({
        'font.size': 14,           # Base font size
        'axes.labelsize': 14,      # Axis label font size
        'xtick.labelsize': 14,     # X-axis tick label size
        'ytick.labelsize': 14,     # Y-axis tick label size
        'legend.fontsize': 14      # Legend font size
    })

    # Define timepoint order
    timepoint_order = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']

    # Define all four regions with styling
    region_configs = [
        ('ant', 'Craniectomy Anterior', 'blue', '-'),           # solid blue
        ('post', 'Craniectomy Posterior', 'red', '-'),          # solid red
        ('baseline_ant', 'Control Anterior', 'blue', '--'),     # dashed blue
        ('baseline_post', 'Control Posterior', 'red', '--')     # dashed red
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect all timepoints present across all regions
    all_timepoints_present = set()

    # First pass: collect all summaries and determine x-axis
    region_summaries = []
    for region, label, color, linestyle in region_configs:
        # Calculate mean WM proportion
        prop_cols = [f'WM_prop_{region}_ring_{ring}' for ring in rings]
        df[f'WM_prop_{region}_mean'] = df[prop_cols].mean(axis=1)

        # Create categorical timepoint
        df['timepoint_cat'] = pd.Categorical(
            df['timepoint'],
            categories=timepoint_order,
            ordered=True
        )

        # Calculate mean and SEM by timepoint
        summary = df.groupby('timepoint_cat')[f'WM_prop_{region}_mean'].agg(['mean', 'sem', 'count']).reset_index()
        summary = summary.dropna()
        region_summaries.append((region, label, color, linestyle, summary))
        all_timepoints_present.update(summary['timepoint_cat'].tolist())

    # Create x-axis based on all timepoints present
    timepoints_present = [tp for tp in timepoint_order if tp in all_timepoints_present]
    x_pos_map = {tp: i for i, tp in enumerate(timepoints_present)}

    # Second pass: plot each region
    for region, label, color, linestyle, summary in region_summaries:
        # Map timepoints to x positions
        x_pos = [x_pos_map[tp] for tp in summary['timepoint_cat']]

        # Plot mean trajectory
        ax.errorbar(x_pos, summary['mean'], yerr=summary['sem'],
                   marker='o', markersize=8, linewidth=2, capsize=5,
                   label=label, color=color, linestyle=linestyle, alpha=0.8)

        # Add individual patient trajectories (faint)
        df['timepoint_cat'] = pd.Categorical(
            df['timepoint'],
            categories=timepoint_order,
            ordered=True
        )
        for patient_id in df['patient_id'].unique():
            patient_data = df[df['patient_id'] == patient_id].sort_values('timepoint_cat')
            if len(patient_data) >= 2:
                tp_indices = [x_pos_map[tp] for tp in patient_data['timepoint'] if tp in x_pos_map]
                wm_vals = patient_data[f'WM_prop_{region}_mean'].values[:len(tp_indices)]
                if len(tp_indices) == len(wm_vals):
                    ax.plot(tp_indices, wm_vals, alpha=0.1, linewidth=1,
                           color=color, linestyle=linestyle)

    # Formatting
    ax.set_xticks(range(len(timepoints_present)))
    ax.set_xticklabels(timepoints_present, rotation=45, ha='right')
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('WM Proportion (Mean Rings 5-7)')
    ax.set_ylim(0.15, 0.8)

    # Legend with larger font in upper right
    ax.legend(fontsize=14, frameon=True, loc='upper right')

    plt.tight_layout()

    # Save to results directory (PNG)
    output_file = os.path.join(output_dir, 'wm_proportion_trajectory_all_regions.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"All regions trajectory plot saved to: {output_file}")

    # Save to thesis directory (PDF, high resolution)
    thesis_dir = '../Thesis/phd-thesis-template-2.4/Chapter6/Figs'
    if os.path.exists(thesis_dir):
        thesis_file = os.path.join(thesis_dir, 'wm_proportion_trajectory_all_regions.pdf')
        plt.savefig(thesis_file, dpi=300, bbox_inches='tight', format='pdf')
        print(f"All regions trajectory plot saved to thesis: {thesis_file}")

    plt.close()

def plot_combined_trajectories(df, rings=[5, 6, 7], output_dir='DTI_Processing_Scripts/wm_proportion_lme_results'):
    """
    Create combined trajectory plots:
    1. Baseline regions: baseline_ant (blue) + baseline_post (red)
    2. Craniectomy regions: ant (blue) + post (red)
    """
    # Set publication style
    set_publication_style()

    # Match jt-test plot font sizes
    plt.rcParams.update({
        'font.size': 14,           # Base font size
        'axes.labelsize': 14,      # Axis label font size
        'xtick.labelsize': 14,     # X-axis tick label size
        'ytick.labelsize': 14,     # Y-axis tick label size
        'legend.fontsize': 14      # Legend font size
    })

    # Define timepoint order
    timepoint_order = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']

    # Define region pairs and colors
    plot_configs = [
        {
            'regions': [('baseline_ant', 'Control Anterior', 'blue'),
                       ('baseline_post', 'Control Posterior', 'red')],
            'filename': 'wm_proportion_trajectory_combined_baseline',
            'ylabel': 'WM Proportion (Mean Rings 5-7)'
        },
        {
            'regions': [('ant', 'Craniectomy Anterior', 'blue'),
                       ('post', 'Craniectomy Posterior', 'red')],
            'filename': 'wm_proportion_trajectory_combined_craniectomy',
            'ylabel': 'WM Proportion (Mean Rings 5-7)'
        }
    ]

    for config in plot_configs:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Collect all timepoints present across both regions
        all_timepoints_present = set()

        # First pass: collect all summaries and determine x-axis
        region_summaries = []
        for region, label, color in config['regions']:
            # Calculate mean WM proportion
            prop_cols = [f'WM_prop_{region}_ring_{ring}' for ring in rings]
            df[f'WM_prop_{region}_mean'] = df[prop_cols].mean(axis=1)

            # Create categorical timepoint
            df['timepoint_cat'] = pd.Categorical(
                df['timepoint'],
                categories=timepoint_order,
                ordered=True
            )

            # Calculate mean and SEM by timepoint
            summary = df.groupby('timepoint_cat')[f'WM_prop_{region}_mean'].agg(['mean', 'sem', 'count']).reset_index()
            summary = summary.dropna()
            region_summaries.append((region, label, color, summary))
            all_timepoints_present.update(summary['timepoint_cat'].tolist())

        # Create x-axis based on all timepoints present
        timepoints_present = [tp for tp in timepoint_order if tp in all_timepoints_present]
        x_pos_map = {tp: i for i, tp in enumerate(timepoints_present)}

        # Second pass: plot each region
        for region, label, color, summary in region_summaries:
            # Map timepoints to x positions
            x_pos = [x_pos_map[tp] for tp in summary['timepoint_cat']]

            # Plot mean trajectory
            ax.errorbar(x_pos, summary['mean'], yerr=summary['sem'],
                       marker='o', markersize=8, linewidth=2, capsize=5,
                       label=label, color=color, alpha=0.8)

            # Add individual patient trajectories (faint)
            df['timepoint_cat'] = pd.Categorical(
                df['timepoint'],
                categories=timepoint_order,
                ordered=True
            )
            for patient_id in df['patient_id'].unique():
                patient_data = df[df['patient_id'] == patient_id].sort_values('timepoint_cat')
                if len(patient_data) >= 2:
                    tp_indices = [x_pos_map[tp] for tp in patient_data['timepoint'] if tp in x_pos_map]
                    wm_vals = patient_data[f'WM_prop_{region}_mean'].values[:len(tp_indices)]
                    if len(tp_indices) == len(wm_vals):
                        ax.plot(tp_indices, wm_vals, alpha=0.1, linewidth=1, color=color)

        # Formatting
        ax.set_xticks(range(len(timepoints_present)))
        ax.set_xticklabels(timepoints_present, rotation=45, ha='right')
        ax.set_xlabel('Timepoint')
        ax.set_ylabel(config['ylabel'])
        ax.set_ylim(0.15, 0.8)

        # Legend with larger font
        ax.legend(fontsize=14, frameon=True, loc='upper right')

        # No title as requested

        plt.tight_layout()

        # Save to results directory (PNG)
        output_file = os.path.join(output_dir, f"{config['filename']}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Combined trajectory plot saved to: {output_file}")

        # Save to thesis directory (PDF, high resolution)
        thesis_dir = '../Thesis/phd-thesis-template-2.4/Chapter6/Figs'
        if os.path.exists(thesis_dir):
            thesis_file = os.path.join(thesis_dir, f"{config['filename']}.pdf")
            plt.savefig(thesis_file, dpi=300, bbox_inches='tight', format='pdf')
            print(f"Combined trajectory plot saved to thesis: {thesis_file}")

        plt.close()

def plot_trajectories_by_anatomical_region(df, rings=[5, 6, 7], output_dir='DTI_Processing_Scripts/wm_proportion_lme_results'):
    """
    Create trajectory plots split by anatomical region (instead of by group):
    1. Anterior region: baseline_ant (control, blue) + ant (craniectomy, red)
    2. Posterior region: baseline_post (control, blue) + post (craniectomy, red)
    """
    # Set publication style
    set_publication_style()

    # Match jt-test plot font sizes
    plt.rcParams.update({
        'font.size': 14,           # Base font size
        'axes.labelsize': 14,      # Axis label font size
        'xtick.labelsize': 14,     # X-axis tick label size
        'ytick.labelsize': 14,     # Y-axis tick label size
        'legend.fontsize': 14      # Legend font size
    })

    # Define timepoint order
    timepoint_order = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']

    # Define region pairs by anatomical location
    plot_configs = [
        {
            'regions': [('baseline_ant', 'Control', 'blue'),
                       ('ant', 'Craniectomy', 'red')],
            'filename': 'wm_proportion_trajectory_anterior_by_group',
            'title': 'Anterior Region',
            'ylabel': 'WM Proportion (Mean Rings 5-7)'
        },
        {
            'regions': [('baseline_post', 'Control', 'blue'),
                       ('post', 'Craniectomy', 'red')],
            'filename': 'wm_proportion_trajectory_posterior_by_group',
            'title': 'Posterior Region',
            'ylabel': 'WM Proportion (Mean Rings 5-7)'
        }
    ]

    for config in plot_configs:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Collect all timepoints present across both regions
        all_timepoints_present = set()

        # First pass: collect all summaries and determine x-axis
        region_summaries = []
        for region, label, color in config['regions']:
            # Calculate mean WM proportion
            prop_cols = [f'WM_prop_{region}_ring_{ring}' for ring in rings]
            df[f'WM_prop_{region}_mean'] = df[prop_cols].mean(axis=1)

            # Create categorical timepoint
            df['timepoint_cat'] = pd.Categorical(
                df['timepoint'],
                categories=timepoint_order,
                ordered=True
            )

            # Calculate mean and SEM by timepoint
            summary = df.groupby('timepoint_cat')[f'WM_prop_{region}_mean'].agg(['mean', 'sem', 'count']).reset_index()
            summary = summary.dropna()
            region_summaries.append((region, label, color, summary))
            all_timepoints_present.update(summary['timepoint_cat'].tolist())

        # Create x-axis based on all timepoints present
        timepoints_present = [tp for tp in timepoint_order if tp in all_timepoints_present]
        x_pos_map = {tp: i for i, tp in enumerate(timepoints_present)}

        # Second pass: plot each region
        for region, label, color, summary in region_summaries:
            # Map timepoints to x positions
            x_pos = [x_pos_map[tp] for tp in summary['timepoint_cat']]

            # Plot mean trajectory
            ax.errorbar(x_pos, summary['mean'], yerr=summary['sem'],
                       marker='o', markersize=8, linewidth=2, capsize=5,
                       label=label, color=color, alpha=0.8)

            # Add individual patient trajectories (faint)
            df['timepoint_cat'] = pd.Categorical(
                df['timepoint'],
                categories=timepoint_order,
                ordered=True
            )
            for patient_id in df['patient_id'].unique():
                patient_data = df[df['patient_id'] == patient_id].sort_values('timepoint_cat')
                if len(patient_data) >= 2:
                    tp_indices = [x_pos_map[tp] for tp in patient_data['timepoint'] if tp in x_pos_map]
                    wm_vals = patient_data[f'WM_prop_{region}_mean'].values[:len(tp_indices)]
                    if len(tp_indices) == len(wm_vals):
                        ax.plot(tp_indices, wm_vals, alpha=0.1, linewidth=1, color=color)

        # Formatting
        ax.set_xticks(range(len(timepoints_present)))
        ax.set_xticklabels(timepoints_present, rotation=45, ha='right')
        ax.set_xlabel('Timepoint')
        ax.set_ylabel(config['ylabel'])
        ax.set_ylim(0.15, 0.8)

        # Legend with larger font
        ax.legend(fontsize=14, frameon=True, loc='upper right')

        # Optional title (uncomment if desired)
        # ax.set_title(config['title'])

        plt.tight_layout()

        # Save to results directory (PNG)
        output_file = os.path.join(output_dir, f"{config['filename']}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Anatomical region trajectory plot saved to: {output_file}")

        # Save to thesis directory (PDF, high resolution)
        thesis_dir = '../Thesis/phd-thesis-template-2.4/Chapter6/Figs'
        if os.path.exists(thesis_dir):
            thesis_file = os.path.join(thesis_dir, f"{config['filename']}.pdf")
            plt.savefig(thesis_file, dpi=300, bbox_inches='tight', format='pdf')
            print(f"Anatomical region trajectory plot saved to thesis: {thesis_file}")

        plt.close()

def main():
    """Main analysis function."""

    print("="*80)
    print("WM PROPORTION LONGITUDINAL STATISTICAL ANALYSIS")
    print("Testing Null Hypothesis: WM proportion does NOT change over time")
    print("="*80)

    # Create output directory
    output_dir = 'DTI_Processing_Scripts/wm_proportion_lme_results'
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    input_file = 'DTI_Processing_Scripts/results/all_wm_proportion_analysis_10x4vox_NEW_filtered.csv'

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)

    # Process timepoints
    df = process_timepoint_data(df)

    print(f"Total observations: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"Timepoints: {sorted(df['timepoint'].unique())}")

    # Analyze each region
    regions = ['ant', 'post', 'baseline_ant', 'baseline_post']
    rings = [5, 6, 7]  # Rings used in longitudinal analysis

    results_dict = {}

    for region in regions:
        results = analyze_region(df, region, rings, output_dir)
        results_dict[region] = results

        # Create trajectory plot
        plot_trajectories_by_region(df, region, rings, output_dir)

    # Create combined trajectory plots
    print(f"\n{'='*80}")
    print("CREATING COMBINED TRAJECTORY PLOTS")
    print(f"{'='*80}")
    plot_combined_trajectories(df, rings, output_dir)

    # Create anatomical region trajectory plots (anterior and posterior separately)
    print(f"\n{'='*80}")
    print("CREATING ANATOMICAL REGION TRAJECTORY PLOTS")
    print(f"{'='*80}")
    plot_trajectories_by_anatomical_region(df, rings, output_dir)

    # Create all-in-one trajectory plot
    print(f"\n{'='*80}")
    print("CREATING ALL REGIONS TRAJECTORY PLOT")
    print(f"{'='*80}")
    plot_trajectories_all(df, rings, output_dir)

    # Create summary table
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL REGIONS")
    print(f"{'='*80}")

    summary_df = create_summary_table(results_dict, output_dir)
    print(f"\n{summary_df.to_string(index=False)}")

    # Overall conclusion
    print(f"\n{'='*80}")
    print("OVERALL CONCLUSION")
    print(f"{'='*80}")

    # Check if summary_df has data
    if summary_df.empty or 'Any_significant' not in summary_df.columns:
        print("✗ Analysis failed - could not fit LME models")
        print("  → Check data quality and sample size")
        return

    sig_regions = summary_df[summary_df['Any_significant'] == 'Yes']['Region'].tolist()
    nonsig_regions = summary_df[summary_df['Any_significant'] == 'No']['Region'].tolist()

    if len(nonsig_regions) == len(regions):
        print("✓ WM proportion did NOT change significantly over time in ANY region")
        print("  → This RULES OUT tissue composition changes as a confound")
        print("  → FA/MD changes reflect true microstructural changes, not tissue composition")
    elif len(sig_regions) == len(regions):
        print("✗ WM proportion changed significantly over time in ALL regions")
        print("  → Requires further analysis to determine if this explains FA/MD patterns")
        print("  → Proceed to Analysis 2: Compare temporal patterns and correlations")
    else:
        print(f"⚠ Mixed results:")
        if sig_regions:
            print(f"  Significant changes: {', '.join(sig_regions)}")
        if nonsig_regions:
            print(f"  No significant changes: {', '.join(nonsig_regions)}")
        print("  → Compare craniectomy (ant/post) vs baseline patterns")

    print(f"\nAll results saved to: {output_dir}/")
    print("Analysis complete!")

if __name__ == '__main__':
    main()
