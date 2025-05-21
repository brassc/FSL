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


def get_clinical_info(patient_id, clinical_df):
    """
    Get sex and age_at_injury for a patient based on their ID format.
    """
    patient_id = str(patient_id)
    
    if len(patient_id) == 5:
        matches = clinical_df[clinical_df['Master_subject_ID'] == patient_id]
    elif len(patient_id) >= 7:
        matches = clinical_df[clinical_df['GUPI'] == patient_id]
    else:
        return None, None
    
    if not matches.empty:
        return matches['Sex'].iloc[0], matches['Age_at_injury'].iloc[0]
    return None, None


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
    metrics_df['Sex'] = None
    metrics_df['Age_at_injury'] = None
    
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
            'Days_since_injury': row['Days_since_injury'],
            'Sex': row['Sex'],
            'Age_at_injury': row['Age_at_injury']
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
            sex = first_row['Sex']
            age_at_injury = first_row['Age_at_injury']
            
            # Update all rows for this patient
            for idx, row in metrics_df[metrics_df['patient_id'] == patient_id].iterrows():
                metrics_df.at[idx, 'Cohort'] = cohort
                metrics_df.at[idx, 'Site'] = site
                metrics_df.at[idx, 'Model'] = model
                metrics_df.at[idx, 'Days_since_injury'] = days_since_injury
                metrics_df.at[idx, 'Sex'] = sex
                metrics_df.at[idx, 'Age_at_injury'] = age_at_injury
    
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
            metrics_df.at[idx, 'Sex'] = mapping_dict[key]['Sex']
            metrics_df.at[idx, 'Age_at_injury'] = mapping_dict[key]['Age_at_injury']
    
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

def process_metrics_file(input_filename, harmonized_output_filename, mean_only=True):
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

    # print(f"patient_scanner_data sample: \n{patient_scanner_data.head()}")
    
    # Also get age @ injury and sex data from Sophies clinical database
    clinical_data= pd.read_csv('Sophie_Data/Sophies_clinical_database_20220822.csv')
    # filter by patient_id


    # Add Sex and Age_at_injury columns to patient_scanner_data
    patient_scanner_data['Sex'] = None
    patient_scanner_data['Age_at_injury'] = None

    # Fill in the clinical data
    for idx, row in patient_scanner_data.iterrows():
        sex, age = get_clinical_info(row['patient_id'], clinical_data)
        patient_scanner_data.at[idx, 'Sex'] = sex
        patient_scanner_data.at[idx, 'Age_at_injury'] = age

    # Print stats on matching
    matched_count = patient_scanner_data['Sex'].notnull().sum()
    total_count = len(patient_scanner_data)
    print(f"Found clinical data for {matched_count} out of {total_count} patients ({matched_count/total_count:.1%})")

    # print(f"patient scanner data example: \n{patient_scanner_data.head()}")
    

    
    


    # # load the metrics data from input file
    all_metrics = pd.read_csv(input_filename)

    # # Convert FA values of 0.0 to NaN
    # fa_columns = [col for col in all_metrics.columns if 'fa' in col.lower()]
    # for column in fa_columns:
    #     # This will catch true zeros and values very close to zero
    #     all_metrics.loc[all_metrics[column] < 0.0001, column] = np.nan
        
    #     # Optional: Print how many zeros were replaced in each column
    #     num_replaced = all_metrics[column].isna().sum()
    #     print(f"Replaced {num_replaced} values with NaN in column {column}")


    pre_harmonised_filename=harmonized_output_filename.replace('_harmonised.csv', '.csv')
    

    all_metrics_merged = merge_scanner_info_with_metrics(
        all_metrics, 
        patient_scanner_data, 
        pre_harmonised_filename
    )
    print(f"all_metrics_merged columns: {all_metrics_merged.columns}")
    print(f"sample data: \n{all_metrics_merged.head()}")
    # Check for missing values in the merged data
    # print(f"\nMissing values in merged data: {all_metrics_merged.isna().sum()}")
    # print(f"list the different categories in all_metrics_merged['Sex']: {all_metrics_merged['Sex'].unique()}")
    # sys.exit()
    
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
    print(f"Missing values in Sex: {all_metrics_merged['Sex'].isna().sum()}")
    print(f"Missing values in Age_at_injury: {all_metrics_merged['Age_at_injury'].isna().sum()}")


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

        if mean_only==True:
            fa_data = valid_data[fa_columns].values.T  # neuroCombat expects features in rows
        else:
            # Create a new DataFrame to hold the extracted values
            numeric_data = pd.DataFrame(index=valid_data.index)

            # Function to extract average from string array
            def extract_mean_from_array_string(val):
                if not isinstance(val, str):
                    return np.nan
                
                # Clean brackets and extract numbers
                val = val.strip()
                if val == '[]' or not val:
                    return np.nan
                    
                # Extract all numbers using regex
                try:
                    numbers = [float(x) for x in re.findall(r'[-+]?\d*\.\d+|\d+', val)]
                    if numbers:
                        return np.mean(numbers)
                    return np.nan
                except:
                    return np.nan
                
            # Process each column
            for col in fa_columns:
                print(f"Processing column {col}...")
                numeric_data[col] = valid_data[col].apply(extract_mean_from_array_string)
                
                # Report stats
                non_nan = numeric_data[col].notna().sum()
                total = len(numeric_data)
                print(f"  - Extracted means for {non_nan} out of {total} values ({non_nan/total:.1%})")
                
                if non_nan > 0:
                    print(f"  - Mean value: {numeric_data[col].mean():.4f}")
            
            # Determine which columns have enough valid data
            valid_fa_columns = []
            for col in fa_columns:
                valid_percent = numeric_data[col].notna().mean() * 100
                percent_threshold=10
                if valid_percent >= percent_threshold:  # Require at least 10% valid values
                    valid_fa_columns.append(col)
                else:
                    print(f"Column {col} has only {valid_percent:.1f}% valid values - excluding (less than {percent_threshold})")
            
            print(f"Using {len(valid_fa_columns)} out of {len(fa_columns)} FA columns")

            if valid_fa_columns:
                # Handle missing values in valid columns by setting NaN to 0
                for col in valid_fa_columns:
                    if numeric_data[col].isna().any():
                        # median = numeric_data[col].median()
                        numeric_data[col] = numeric_data[col].fillna(0.0)
                        print(f"Filled NaN values in {col} with 0.0")
                
                # Extract as numpy array
                fa_data = numeric_data[valid_fa_columns].values.T
                print(f"Final FA data shape: {fa_data.shape}")
                
                # # Verify data is ready for neuroCombat
                # print(f"Final data type: {fa_data.dtype}")
                # print(f"Contains NaN: {np.isnan(fa_data).any()}")
                # print(f"Contains Inf: {np.isinf(fa_data).any()}")
     
        covars_dict = {
            'batch': valid_data['Site_Model'].values,
            'timepoint': valid_data['timepoint'].values,
            'Cohort': valid_data['Cohort'].values,
            'Days_since_injury': valid_data['Days_since_injury'].values,
            'Sex': valid_data['Sex'].values,
            'Age_at_injury': valid_data['Age_at_injury'].values
        }

        covars_df = pd.DataFrame(covars_dict)

        # Apply neuroCombat for FA metrics
        print("Running neuroCombat on FA metrics...")
        fa_combat_data = neuroCombat(
            dat=fa_data,
            covars=covars_df,
            batch_col='batch',
            #categorical_cols=categorical_cols,
            categorical_cols=['Sex'],
            #continuous_cols=None,
            continuous_cols=['Age_at_injury'],
            eb=True,
            parametric=True,
            mean_only=mean_only,
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


    # Extract MD data for harmonisation
    print("\nHarmonizing MD metrics...")
    md_columns = [col for col in valid_data.columns if col.startswith('md_')]
    if len(md_columns) > 0:
        print(f"Found {len(md_columns)} MD metrics")
        
        if mean_only==True:
            md_data = valid_data[md_columns].values.T  # neuroCombat expects features in rows
        else:
            # Create a new DataFrame to hold the extracted values
            numeric_data_md = pd.DataFrame(index=valid_data.index)

            # Function to extract average from string array
            def extract_mean_from_array_string(val):
                if not isinstance(val, str):
                    return np.nan
                
                # Clean brackets and extract numbers
                val = val.strip()
                if val == '[]' or not val:
                    return np.nan
                    
                # Extract all numbers using regex
                try:
                    numbers = [float(x) for x in re.findall(r'[-+]?\d*\.\d+|\d+', val)]
                    if numbers:
                        return np.mean(numbers)
                    return np.nan
                except:
                    return np.nan
                
            # Process each column
            for col in md_columns:
                print(f"Processing column {col}...")
                numeric_data_md[col] = valid_data[col].apply(extract_mean_from_array_string)
                
                # Report stats
                non_nan = numeric_data_md[col].notna().sum()
                total = len(numeric_data_md)
                print(f"  - Extracted means for {non_nan} out of {total} values ({non_nan/total:.1%})")
                
                if non_nan > 0:
                    print(f"  - Mean value: {numeric_data_md[col].mean():.4f}")

            
            # Determine which columns have enough valid data
            valid_md_columns = []
            for col in md_columns:
                valid_percent = numeric_data_md[col].notna().mean() * 100
                percent_threshold=10
                if valid_percent >= percent_threshold:  # Require at least 10% valid values
                    valid_md_columns.append(col)
                else:
                    print(f"Column {col} has only {valid_percent:.1f}% valid values - excluding (less than {percent_threshold})")
            
            print(f"Using {len(valid_md_columns)} out of {len(md_columns)} MD columns")

            if valid_md_columns:
                # Handle missing values in valid columns by setting NaN to 0
                for col in valid_md_columns:
                    if numeric_data_md[col].isna().any():
                        numeric_data_md[col] = numeric_data_md[col].fillna(0.0)
                        print(f"Filled NaN values in {col} with 0.0")
                
                # Extract as numpy array
                md_data = numeric_data_md[valid_md_columns].values.T
                print(f"Final MD data shape: {md_data.shape}")

        
        # Set up covariates following the exact function signature (same as FA harmonization)
        covars_dict = {
            'batch': valid_data['Site_Model'].values,
            'timepoint': valid_data['timepoint'].values,
            'Cohort': valid_data['Cohort'].values,
            'Days_since_injury': valid_data['Days_since_injury'].values,
            'Sex': valid_data['Sex'].values,
            'Age_at_injury': valid_data['Age_at_injury'].values
        }

        covars_df = pd.DataFrame(covars_dict)
        
        # Apply neuroCombat for MD metrics
        print("Running neuroCombat on MD metrics...")
        md_combat_data = neuroCombat(
            dat=md_data,
            covars=covars_df,
            batch_col='batch',
            categorical_cols=None,
            continuous_cols=['Age_at_injury'],
            eb=True,
            parametric=True,
            mean_only=mean_only,
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


    # Clean FA data - make values < 0.0001 NaN
    fa_columns = [col for col in harmonized_data.columns if 'fa' in col.lower()]
    for column in fa_columns:
        # This will catch true zeros and values very close to zero
        harmonized_data.loc[harmonized_data[column] < 0.0001, column] = np.nan
        # Optional: Print how many zeros were replaced in each column
        num_replaced = harmonized_data[column].isna().sum()
        print(f"Replaced {num_replaced} values with NaN in column {column}")

    # Clean MD data - make values < 0.0 NaN
    md_columns = [col for col in harmonized_data.columns if 'md' in col.lower()]
    for column in md_columns:
        # This will catch true zeros and values very close to zero
        harmonized_data.loc[harmonized_data[column] < 0.0, column] = np.nan
        # Optional: Print how many zeros were replaced in each column
        num_replaced = harmonized_data[column].isna().sum()
        print(f"Replaced {num_replaced} values with NaN in column {column}")


    # Get the original file name from the input
    # original_file = merged_filename
    # Create new filename with _harmonised suffix
    #harmonized_output_file = original_file.replace('.csv', '_harmonised.csv')

    # Save harmonized data
    harmonized_data.to_csv(harmonized_output_filename, index=False)


import pandas as pd
import numpy as np
import os
import re

def average_rings(input_filename, output_filename, rings_to_average):
    """
    Process DTI metrics data to average specified rings for each metric-location combination.
    
    Parameters:
    -----------
    input_filename : str
        Path to the input CSV file
    output_filename : str
        Path where the output CSV will be saved
    rings_to_average : list of int
        List of ring numbers to average (e.g., [5, 6, 7])
    
    Returns:
    --------
    pd.DataFrame
        The processed DataFrame with averaged ring values
    """
    # Validate inputs
    if not isinstance(rings_to_average, list) or len(rings_to_average) == 0:
        raise ValueError("rings_to_average must be a non-empty list of integers")
    
    # Read the CSV file
    print(f"Reading input file: {input_filename}")
    df = pd.read_csv(input_filename)
    
    # Create a new DataFrame for results
    result_df = pd.DataFrame()
    
    # Copy patient_id and timepoint columns
    result_df['patient_id'] = df['patient_id']
    result_df['timepoint'] = df['timepoint']
    
    # Convert rings to strings for column naming
    ring_str = '_'.join(str(ring) for ring in sorted(rings_to_average))
    
    # Identify all unique metric+location combinations
    metric_location_patterns = set()
    for col in df.columns:
        match = re.match(r'^([A-Z]+)_([a-z_]+)_ring_\d+', col)
        if match:
            metric_location_patterns.add(f"{match.group(1)}_{match.group(2)}")
    
    print(f"Found {len(metric_location_patterns)} metric-location patterns")
    
    # Process each metric-location pattern
    for pattern in sorted(metric_location_patterns):
        print(f"Processing pattern: {pattern}")
        
        # Find the columns for specified rings
        ring_cols = [f"{pattern}_ring_{ring}" for ring in rings_to_average]
        
        # New column name for the average
        result_col_name = f"{pattern}_ring_{ring_str}_avg"
        
        # Process each row
        result_column = []
        for idx, row in df.iterrows():
            # Initialize list to store all values from specified rings
            all_values = []
            
            # Process each ring column if it exists
            for ring_col in ring_cols:
                if ring_col in df.columns:
                    try:
                        value = row[ring_col]
                        if isinstance(value, str):
                            # Extract all floating point numbers from the string
                            floats_array = [float(x) for x in re.findall(r'[-+]?\d*\.\d+|\d+', value)]
                        else:
                            floats_array = [float(value)] if not pd.isna(value) else []
                        
                        # Apply filtering for FA columns only
                        if pattern.startswith('FA_'):
                            filtered_array = [x for x in floats_array if 0.05 <= x <= 0.8]
                            # Use original if filtering removes all values
                            if not filtered_array:
                                filtered_array = floats_array
                        else:  # For MD columns
                            filtered_array = floats_array
                        
                        # Add values to the combined list
                        all_values.extend(filtered_array)
                    except (ValueError, SyntaxError, TypeError) as e:
                        print(f"Error processing {ring_col} in row {idx}: {e}")
                        # If there's an error, skip this ring
                        continue
            
            # Calculate mean of combined values
            if all_values:
                mean_value = np.mean(all_values)
            else:
                mean_value = np.nan
            
            result_column.append(mean_value)
        
        # Add the result column to the result DataFrame
        result_df[result_col_name] = result_column
        print(f"Added column: {result_col_name}")
    
    # Save the result to CSV
    result_df.to_csv(output_filename, index=False, float_format='%.10f')
    print(f"Processing complete. Output saved to {output_filename}")
    
    return result_df




################################
# Usage
####################################

# # process_metrics_file(input_filename='DTI_Processing_Scripts/results/all_metrics_5x4vox.csv', 
# #                      harmonized_output_filename='DTI_Processing_Scripts/merged_data_5x4vox_harmonised.csv')

# # process_metrics_file(input_filename='DTI_Processing_Scripts/results/all_metrics_5x4vox_NEW.csv',
# #                      harmonized_output_filename='DTI_Processing_Scripts/merged_data_5x4vox_NEW_harmonised.csv')


# # process_metrics_file(input_filename='DTI_Processing_Scripts/results/all_metrics_5x4vox_NEW_filtered.csv',
#                     #  harmonized_output_filename='DTI_Processing_Scripts/merged_data_5x4vox_NEW_filtered_harmonised.csv')

# # process_metrics_file(input_filename='DTI_Processing_Scripts/results/all_metrics_5x4vox_NEW_filtered_wm.csv',
# #                      harmonized_output_filename='DTI_Processing_Scripts/merged_data_5x4vox_NEW_filtered_wm_harmonised.csv')

# process_metrics_file(input_filename='DTI_Processing_Scripts/results/all_metrics_10x4vox_NEW_filtered_wm.csv',
#                      harmonized_output_filename='DTI_Processing_Scripts/merged_data_10x4vox_NEW_filtered_wm_harmonised.csv')
                

# # process_metrics_file(input_filename='DTI_Processing_Scripts/results/all_metrics_10x4vox_NEW_filtered.csv',
# #                      harmonized_output_filename='DTI_Processing_Scripts/merged_data_10x4vox_NEW_filtered_harmonised.csv')



#### MERGING AND AVERAGING OF SPECIFIED RINGS ####

wm_all_values_filename='DTI_Processing_Scripts/results/all_metrics_10x4vox_NEW_filtered_all_values_wm.csv'
output_wm_rings_567_filename = 'DTI_Processing_Scripts/results/all_metrics_10x4vox_NEW_filtered_rings_5_6_7_mean_wm.csv'

# processed_wm_rings_567_filename_harmonised = "DTI_Processing_Scripts/merged_data_10x4vox_filtered_wm_rings_567_harmonised.csv"
# output_wm_rings_567_filename_harmonised = 'DTI_Processing_Scripts/results/all_metrics_10x4vox_NEW_filtered_rings_5_6_7_mean_wm_harmonised.csv'

    
average_rings(wm_all_values_filename, output_wm_rings_567_filename, rings= [5, 6, 7])

process_metrics_file(input_filename=output_wm_rings_567_filename, 
                     harmonized_output_filename="DTI_Processing_Scripts/merged_data_10x4vox_NEW_filtered_wm_567_harmonised.csv")

# process_metrics_file(input_filename=wm_all_values_filename, 
#                      harmonized_output_filename=processed_wm_rings_567_filename_harmonised,
#                      mean_only=False)
# # spits out average anyway - so just do average first.



print("\n\nHarmonization complete!")

