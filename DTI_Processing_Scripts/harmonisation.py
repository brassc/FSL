import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuroCombat import neuroCombat
import sys
import patsy

def check_singularities(valid_data):
    """
    Check for singularity issues in the data, particularly related to timepoints within Site_Model.
    
    Args:
        valid_data: The DataFrame to analyze
        
    Returns:
        bool: Whether there is a risk of singularity
    """
    # Check for singleton timepoints within each Site_Model
    singleton_timepoints_count = 0
    total_site_model_timepoint_combinations = 0
    site_models_with_singletons = 0

    print("\nAnalyzing timepoint distribution within each Site_Model:")
    for sm in valid_data['Site_Model'].unique():
        sm_data = valid_data[valid_data['Site_Model'] == sm]
        timepoint_counts = sm_data['timepoint'].value_counts()
        
        # Count singletons (timepoints with only 1 sample) in this Site_Model
        singletons_in_this_sm = sum(timepoint_counts == 1)
        total_timepoints_in_this_sm = len(timepoint_counts)
        
        if singletons_in_this_sm > 0:
            site_models_with_singletons += 1
            
        singleton_timepoints_count += singletons_in_this_sm
        total_site_model_timepoint_combinations += total_timepoints_in_this_sm
        
        print(f"\nSite_Model: {sm} ({len(sm_data)} samples)")
        print(f"Total unique timepoints: {total_timepoints_in_this_sm}")
        print(f"Timepoints with only 1 sample: {singletons_in_this_sm}")
        print(f"Singleton percentage: {singletons_in_this_sm/total_timepoints_in_this_sm*100:.1f}%")
        
        # List the singleton timepoints
        if singletons_in_this_sm > 0:
            singleton_list = timepoint_counts[timepoint_counts == 1].index.tolist()
            print(f"Singleton timepoints: {', '.join(singleton_list)}")

    # Summary statistics
    print("\nSummary:")
    print(f"Total Site_Model combinations: {len(valid_data['Site_Model'].unique())}")
    print(f"Site_Model combinations with singleton timepoints: {site_models_with_singletons}")
    print(f"Total unique Site_Model-timepoint combinations: {total_site_model_timepoint_combinations}")
    print(f"Total singleton timepoints: {singleton_timepoints_count}")
    print(f"Singleton percentage overall: {singleton_timepoints_count/total_site_model_timepoint_combinations*100:.1f}%")

    # Risk assessment
    singularity_risk = False
    if singleton_timepoints_count > 0:
        print("\nRisk of singularity detected! Some timepoints within Site_Model combinations have only one sample.")
        print("This causes a singularity problem when using timepoint as a categorical covariate.")
        print("Consider using a continuous variable (Days_since_injury) instead of categorical timepoint,")
        print("or use only batch without any covariates in the harmonization.")
        singularity_risk = True
        
    return singularity_risk

def merge_scanner_info_with_metrics(metrics_df, scanner_info_df, output_filename):
    """
    Merge scanner information (Cohort, Site, Model) with metrics data.
    
    Args:
        metrics_df: DataFrame containing metrics data
        scanner_info_df: DataFrame containing scanner information
        output_filename: Path to save the merged data
        
    Returns:
        DataFrame: The merged data with scanner information added
    """
    # Clean data from spaces everywhere
    metrics_df.columns = metrics_df.columns.str.strip().str.replace(' ', '_').str.replace('?', '').str.replace('(', '').str.replace(')', '').str.lower()
    metrics_df = metrics_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    metrics_df['patient_id'] = metrics_df['patient_id'].astype(str)
    metrics_df['timepoint'] = metrics_df['timepoint'].astype(str)
    
    # Add new columns for scanner information
    metrics_df['Cohort'] = None
    metrics_df['Site'] = None
    metrics_df['Model'] = None
    metrics_df['Days_since_injury'] = None
    
    # Create a mapping dictionary for 5-digit patient IDs
    mapping_dict = {}
    for idx, row in scanner_info_df[scanner_info_df['patient_id'].astype(str).str.len() == 5].iterrows():
        patient_id = str(row['patient_id'])
        timepoint = str(row['timepoint'])
        key = (patient_id, timepoint)
        mapping_dict[key] = {
            'Cohort': row['Cohort'],
            'Site': row['Site'],
            'Model': row['Model'],
            'Days_since_injury': row['Days_since_injury']
        }
    
    # Process 7-digit patient IDs directly
    seven_digit_patients = scanner_info_df[scanner_info_df['patient_id'].astype(str).str.len() == 7]['patient_id'].unique()
    for patient_id in seven_digit_patients:
        patient_rows = scanner_info_df[scanner_info_df['patient_id'] == patient_id]
        if not patient_rows.empty:
            first_row = patient_rows.iloc[0]
            cohort = first_row['Cohort']
            site = first_row['Site']
            model = first_row['Model']
            days_since_injury = first_row['Days_since_injury']
            
            # Update all rows for this patient
            for idx, row in metrics_df[metrics_df['patient_id'] == patient_id].iterrows():
                metrics_df.at[idx, 'Cohort'] = cohort
                metrics_df.at[idx, 'Site'] = site
                metrics_df.at[idx, 'Model'] = model
                metrics_df.at[idx, 'Days_since_injury'] = days_since_injury
    
    # Update 5-digit patient IDs using the mapping dictionary
    for idx, row in metrics_df[metrics_df['patient_id'].astype(str).str.len() == 5].iterrows():
        patient_id = str(row['patient_id'])
        timepoint = str(row['timepoint'])
        key = (patient_id, timepoint)
        if key in mapping_dict:
            metrics_df.at[idx, 'Cohort'] = mapping_dict[key]['Cohort']
            metrics_df.at[idx, 'Site'] = mapping_dict[key]['Site']
            metrics_df.at[idx, 'Model'] = mapping_dict[key]['Model']
            metrics_df.at[idx, 'Days_since_injury'] = mapping_dict[key]['Days_since_injury']
    
    # Report results
    updated_count = metrics_df[metrics_df['Cohort'].notnull()].shape[0]
    print(f"Updated {updated_count} out of {metrics_df.shape[0]} rows")

    # Check Days_since_injury stats
    days_count = metrics_df['Days_since_injury'].notnull().sum()
    print(f"Days_since_injury data available for {days_count} out of {metrics_df.shape[0]} rows")
    
    
    # Save to file
    metrics_df.to_csv(output_filename, index=False)
    print(f"\nMerged data saved to {output_filename}")
    
    return metrics_df

def process_metrics_file(input_filename, harmonized_output_filename):
    """
    Process a metrics file through the entire pipeline.
    
    Args:
        input_filename: Path to the metrics CSV file (e.g., all_metrics_5x4vox.csv)
    """
    # load data 
    # Load Sophie's scan database
    scandataloc = 'Sophie_Data/Sophies_scan_database_20220822.csv'
    scandata = pd.read_csv(scandataloc)
    #print(scandata.columns)

    # import patient scanner data
    patient_scanner_data = pd.read_csv('DTI_Processing_Scripts/patient_scanner_data_with_timepoints.csv')
    # tidy it
    patient_scanner_data = patient_scanner_data.dropna(subset=['timepoint'])
    print(f"Total entries in patient_scanner_data: {len(patient_scanner_data)}")
    # get unique patient id and timepoint combinations
    patient_scanner_data = patient_scanner_data.drop_duplicates(subset=['patient_id', 'timepoint'])
    print(f"Unique patient_id and timepoint combinations: {len(patient_scanner_data)}")




    # # load the metrics data from input file
    all_metrics = pd.read_csv(input_filename)
    pre_harmonised_filename=harmonized_output_filename.replace('_harmonised.csv', '.csv')
    

    all_metrics_merged = merge_scanner_info_with_metrics(
        all_metrics, 
        patient_scanner_data, 
        pre_harmonised_filename
    )
    # return

    ## HARMONISATION
    # Extract FA data for harmonisation
    print("\nHarmonizing FA metrics...")
    # Create a combined Site_Model batch variable
    all_metrics_merged['Site_Model'] = all_metrics_merged['Site'].astype(str) + '_' + all_metrics_merged['Model'].astype(str)
    print(f"\nUnique Site_Model combinations: {all_metrics_merged['Site_Model'].unique()}")



    print(f"\nMissing values in Site_Model: {all_metrics_merged['Site_Model'].isna().sum()}")
    print(f"Missing values in timepoint: {all_metrics_merged['timepoint'].isna().sum()}")
    print(f"Missing values in Cohort: {all_metrics_merged['Cohort'].isna().sum()}")
    print(f"Missing values in days_since_injury: {all_metrics_merged['Days_since_injury'].isna().sum()}")


    # # Filter out rows with missing batch variable
    valid_data = all_metrics_merged.dropna(subset=['Site_Model'])
    print(f"\nRows after removing missing Site_Model: {len(valid_data)} (removed {len(all_metrics_merged) - len(valid_data)} rows)")


    check_singularities(valid_data)

    #print(f"valid data columns: {valid_data.columns}")

    # mod matrix / R like implementation
    # formula = "~ C(timepoint) + C(Cohort)"
    # mod = patsy.dmatrix(formula, data=valid_data)
    # print(f"\nCreated model matrix with {mod.shape[1]} variables")
    # print(f"Model matrix: \n{mod}")



    # Extract FA data for harmonisation
    print("\nHarmonizing FA metrics...")
    fa_columns = [col for col in valid_data.columns if col.startswith('fa_')]
    if len(fa_columns) > 0:
        print(f"Found {len(fa_columns)} FA metrics")
        fa_data = valid_data[fa_columns].values.T  # neuroCombat expects features in rows
        
        # Set up covariates following the exact function signature: 
        # neuroCombat(dat: Any, covars: Any, batch_col: Any, categorical_cols: Any | None = None, 
        #             continuous_cols: Any | None = None, eb: bool = True, parametric: bool = True, 
        #             mean_only: bool = False, ref_batch: Any | None = None) -> dict[str, Any]
        
        covars_dict = {
            'batch': valid_data['Site_Model'].values,
            'timepoint': valid_data['timepoint'].values,
            'Cohort': valid_data['Cohort'].values,
            'Days_since_injury': valid_data['Days_since_injury'].values
        }

        covars_df = pd.DataFrame(covars_dict)

        
        
        # Apply neuroCombat for FA metrics
        print("Running neuroCombat on FA metrics...")
        fa_combat_data = neuroCombat(
            dat=fa_data,
            covars=covars_df,
            batch_col='batch',
            #categorical_cols=categorical_cols,
            categorical_cols=None,
            #continuous_cols=None,
            continuous_cols=['Days_since_injury'],
            eb=True,
            parametric=True,
            mean_only=True,
            ref_batch=None
        )
        
        # Create FA harmonized dataframe
        fa_harmonized_df = pd.DataFrame(
            fa_combat_data['data'].T,  # Transpose back to original orientation
            columns=[f"harmonized_{col}" for col in fa_columns],
            index=valid_data.index
        )
    else:
        print("No FA columns found!")
        fa_harmonized_df = pd.DataFrame(index=valid_data.index)





    # Add this code after the FA harmonization section in your script

    # Extract MD data for harmonisation
    print("\nHarmonizing MD metrics...")
    md_columns = [col for col in valid_data.columns if col.startswith('md_')]
    if len(md_columns) > 0:
        print(f"Found {len(md_columns)} MD metrics")
        md_data = valid_data[md_columns].values.T  # neuroCombat expects features in rows
        
        # Set up covariates following the exact function signature (same as FA harmonization)
        covars_dict = {
            'batch': valid_data['Site_Model'].values,
            'timepoint': valid_data['timepoint'].values,
            'Cohort': valid_data['Cohort'].values,
            'Days_since_injury': valid_data['Days_since_injury'].values
        }

        covars_df = pd.DataFrame(covars_dict)
        
        # Apply neuroCombat for MD metrics
        print("Running neuroCombat on MD metrics...")
        md_combat_data = neuroCombat(
            dat=md_data,
            covars=covars_df,
            batch_col='batch',
            categorical_cols=None,
            continuous_cols=['Days_since_injury'],
            eb=True,
            parametric=True,
            mean_only=True,
            ref_batch=None
        )
        
        # Create MD harmonized dataframe
        md_harmonized_df = pd.DataFrame(
            md_combat_data['data'].T,  # Transpose back to original orientation
            columns=[f"harmonized_{col}" for col in md_columns],
            index=valid_data.index
        )
    else:
        print("No MD columns found!")
        md_harmonized_df = pd.DataFrame(index=valid_data.index)

    print("md_harmonized_df columns:")
    print(md_harmonized_df.columns)

    # Create a new dataframe that's a copy of the original data
    harmonized_data = valid_data.copy()

    # Replace FA values with harmonized versions
    if len(fa_columns) > 0:
        # Replace original FA columns with harmonized values
        for i, col in enumerate(fa_columns):
            # The harmonized values are already in fa_harmonized_df, we just need to extract them
            harmonized_col_name = f"harmonized_{col}"
            if harmonized_col_name in fa_harmonized_df.columns:
                harmonized_data[col] = fa_harmonized_df[harmonized_col_name].values
        print(f"Replaced {len(fa_columns)} FA columns with their harmonized versions")

    # Replace MD values with harmonized versions
    if len(md_columns) > 0:
        # Replace original MD columns with harmonized values
        for i, col in enumerate(md_columns):
            # The harmonized values are already in md_harmonized_df, we just need to extract them
            harmonized_col_name = f"harmonized_{col}"
            if harmonized_col_name in md_harmonized_df.columns:
                harmonized_data[col] = md_harmonized_df[harmonized_col_name].values
        print(f"Replaced {len(md_columns)} MD columns with their harmonized versions")

    # Get the original file name from the input
    # original_file = merged_filename
    # Create new filename with _harmonised suffix
    #harmonized_output_file = original_file.replace('.csv', '_harmonised.csv')

    # Save harmonized data
    harmonized_data.to_csv(harmonized_output_filename, index=False)

# Usage

# process_metrics_file(input_filename='DTI_Processing_Scripts/results/all_metrics_5x4vox.csv', 
#                      harmonized_output_filename='DTI_Processing_Scripts/merged_data_5x4vox_harmonised.csv')

# process_metrics_file(input_filename='DTI_Processing_Scripts/results/all_metrics_5x4vox_NEW.csv',
#                      harmonized_output_filename='DTI_Processing_Scripts/merged_data_5x4vox_NEW_harmonised.csv')


process_metrics_file(input_filename='DTI_Processing_Scripts/results/all_metrics_5x4vox_NEW_filtered.csv',
                     harmonized_output_filename='DTI_Processing_Scripts/merged_data_5x4vox_NEW_filtered_harmonised.csv')

# process_metrics_file(input_filename='DTI_Processing_Scripts/results/all_metrics_10x4vox_NEW_filtered.csv',
#                      harmonized_output_filename='DTI_Processing_Scripts/merged_data_10x4vox_NEW_filtered_harmonised.csv')

