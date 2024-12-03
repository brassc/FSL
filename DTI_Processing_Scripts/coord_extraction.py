import pandas as pd

# Load .csv
csv_file="/home/cmb247/repos/FSL/Image_Processing_Scripts/included_patient_info.csv"
data = pd.read_csv(csv_file)

# Define the function to generate FSL command for creating ROIs
def generate_fsl_command(row, output_dir):
    patient_id=row['patient ID']
    timepoint = row['timepoint']
    z = row['z coord (slice)']

    anterior_x = row['anterior x coord']
    anterior_y = row['anterior y coord']
    posterior_x = row['posterior x coord']
    posterior_y = row['posterior y coord']

    baseline_anterior_x = row['baseline anterior x coord']
    baseline_posterior_x = row['baseline posterior x coord']
    
    side = row['side (L/R)']

    radius = 5

    # Generate filenames for output
    anterior_roi_file = f"{output_dir}/roi_{patient_id}_{timepoint}_anterior.nii.gz"
    posterior_roi_file = f"{output_dir}/roi_{patient_id}_{timepoint}_posterior.nii.gz"
    baseline_anterior_roi_file = f"{output_dir}/roi_{patient_id}_{timepoint}_baseline_anterior.nii.gz"
    baseline_posterior_roi_file = f"{output_dir}/roi_{patient_id}_{timepoint}_baseline_posterior.nii.gz"

    # FSL command for anterior ROI
    anterior_cmd = (f"fslmaths /path/to/T1_or_FA_data -mul 0 -add 1 -roi {anterior_x} 1 {anterior_y} 1 {z} 1 0 1 "
                    f"-kernel sphere {radius} -fmean {anterior_roi_file}")
    
    # FSL command for posterior ROI
    posterior_cmd = (f"fslmaths /path/to/T1_or_FA_data -mul 0 -add 1 -roi {posterior_x} 1 {posterior_y} 1 {z} 1 0 1 "
                     f"-kernel sphere {radius} -fmean {posterior_roi_file}")

    # FSL command for baseline anterior ROI
    baseline_anterior_cmd = (f"fslmaths /path/to/T1_or_FA_data -mul 0 -add 1 -roi {baseline_anterior_x} 1 {anterior_y} 1 {z} 1 0 1 "
                    f"-kernel sphere {radius} -fmean {baseline_anterior_roi_file}")

    # FSL command for baseline posterior ROI
    baseline_posterior_cmd = (f"fslmaths /path/to/T1_or_FA_data -mul 0 -add 1 -roi {baseline_posterior_x} 1 {posterior_y} 1 {z} 1 0 1 "
                     f"-kernel sphere {radius} -fmean {baseline_posterior_roi_file}")
    
    return anterior_cmd, posterior_cmd, baseline_anterior_cmd, baseline_posterior_cmd


# Output dir for ROIs
output_dir = '/home/cmb247/repos/FSL/DTI_Processing_Scripts'

# Iterate through csv and generate fsl commands for each row 

commands = []
for index, row in data.iterrows():
    if row['excluded?'] == 0:
        anterior_cmd, posterior_cmd, baseline_anterior_cmd, baseline_posterior_cmd = generate_fsl_command(row, output_dir)












