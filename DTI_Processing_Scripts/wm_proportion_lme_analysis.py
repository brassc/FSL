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
from scipy import stats
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# R integration for LME models
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
pandas2ri.activate()

def setup_r_environment():
    """Set up R environment with required packages."""
    try:
        base = importr('base')
        lme4 = importr('lme4')
        lmerTest = importr('lmerTest')
        return True
    except Exception as e:
        print(f"Error loading R packages: {e}")
        print("Please ensure R packages 'lme4' and 'lmerTest' are installed")
        return False

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
    Fit LME model: DV ~ timepoint + (1|patient_id)

    Args:
        df: DataFrame with columns [patient_id, timepoint, dv_column]
        dv_column: Name of dependent variable column
        output_file: Optional file to write R output

    Returns:
        Dictionary with model results
    """
    # Remove NA values
    df_clean = df[['patient_id', 'timepoint', dv_column]].dropna()

    if len(df_clean) < 10:
        print(f"Warning: Only {len(df_clean)} observations for {dv_column}, skipping analysis")
        return None

    # Convert to R dataframe
    r_df = pandas2ri.py2rpy(df_clean)
    ro.globalenv['data'] = r_df

    # Fit LME model
    formula = f'{dv_column} ~ timepoint + (1|patient_id)'

    try:
        # Fit model
        model = ro.r(f'''
            library(lmerTest)
            model <- lmer({formula}, data=data, REML=TRUE)
            model
        ''')

        # Get summary
        summary = ro.r('summary(model)')

        # Extract fixed effects
        fixed_effects = ro.r('fixef(model)')

        # Get ANOVA table (Type III tests)
        anova_table = ro.r('anova(model, type="III")')

        # Extract p-value for timepoint effect
        anova_df = pandas2ri.rpy2py(anova_table)
        timepoint_pval = anova_df.loc['timepoint', 'Pr(>F)'] if 'timepoint' in anova_df.index else np.nan

        # Get coefficients table
        coef_summary = ro.r('summary(model)$coefficients')
        coef_df = pandas2ri.rpy2py(coef_summary)

        # Write detailed output if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"Linear Mixed Effects Model Results\n")
                f.write(f"Formula: {formula}\n")
                f.write(f"=" * 80 + "\n\n")
                f.write(f"ANOVA (Type III tests):\n")
                f.write(anova_df.to_string())
                f.write(f"\n\n")
                f.write(f"Fixed Effects Coefficients:\n")
                f.write(coef_df.to_string())
                f.write(f"\n\n")
                f.write(f"Overall Timepoint Effect p-value: {timepoint_pval:.6f}\n")

        results = {
            'timepoint_pval': timepoint_pval,
            'anova_table': anova_df,
            'coefficients': coef_df,
            'n_obs': len(df_clean),
            'n_patients': df_clean['patient_id'].nunique()
        }

        return results

    except Exception as e:
        print(f"Error fitting LME model for {dv_column}: {e}")
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
        print(f"Overall timepoint effect p-value: {results['timepoint_pval']:.6f}")

        if results['timepoint_pval'] < 0.05:
            print(f"→ SIGNIFICANT: WM proportion changes significantly over time (p < 0.05)")
        else:
            print(f"→ NOT SIGNIFICANT: WM proportion does NOT change significantly over time (p ≥ 0.05)")

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
                'Timepoint_p_value': results['timepoint_pval'],
                'Significant': 'Yes' if results['timepoint_pval'] < 0.05 else 'No'
            })

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    output_file = os.path.join(output_dir, 'wm_proportion_lme_summary.csv')
    summary_df.to_csv(output_file, index=False)
    print(f"\nSummary table saved to: {output_file}")

    return summary_df

def plot_trajectories_by_region(df, region, rings=[5, 6, 7], output_dir='DTI_Processing_Scripts/wm_proportion_lme_results'):
    """Create trajectory plot for a specific region."""
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
    ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
    ax.set_ylabel('WM Proportion (Mean Rings 5-7)', fontsize=12, fontweight='bold')

    region_title = region.replace('_', ' ').title()
    ax.set_title(f'WM Proportion Over Time: {region_title}', fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.legend()

    # Save
    output_file = os.path.join(output_dir, f'wm_proportion_trajectory_{region}.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Trajectory plot saved to: {output_file}")
    plt.close()

def main():
    """Main analysis function."""

    print("="*80)
    print("WM PROPORTION LONGITUDINAL STATISTICAL ANALYSIS")
    print("Testing Null Hypothesis: WM proportion does NOT change over time")
    print("="*80)

    # Check R environment
    if not setup_r_environment():
        sys.exit(1)

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

    sig_regions = summary_df[summary_df['Significant'] == 'Yes']['Region'].tolist()
    nonsig_regions = summary_df[summary_df['Significant'] == 'No']['Region'].tolist()

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
