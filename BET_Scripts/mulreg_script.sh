#!/bin/bash

#RUN FROM "BET_Extractions" DIRECTORY LEVEL

# Define the standard space template path (change this to your standard space template)
standard_template="mni_icbm152_t1_tal_nlin_sym_55_ext.nii"
standard_space_name="mni152"


# List of subdirectory names (e.g., 12519, 13990, 14324, 16577, 20942)
subdirectories=("12519" "13198" "13782" "13990" "14324" "16754" "19344" "19575" "19978" "19981" "20174" "20651" "20942" "21221" "22725" "22785" "23348")

# Loop through each subdirectory
for subdirectory in "${subdirectories[@]}"; do
    input_directory="${subdirectory}/OG_Scans"
    output_directory="${subdirectory}/MNI152_registered_scans"

    # Make sure the output directory exists; if not, create it
    mkdir -p "$output_directory"

    # Loop through the brain scans in the input directory
    for input_scan in "$input_directory"/*.nii.gz; do
        # Extract the scan filename without the path and extension
        scan_name=$(basename "${input_scan%.nii.gz}")

        # Define the output path for the registered scan
        output_scan="$output_directory/$standard_space_name.${scan_name}_registered.nii.gz"

        # Perform registration using flirt or fnirt (adjust parameters as needed)
	echo "Completing registration for $input_scan to $standard_template" >> mulreg_log.txt
        flirt -in "$input_scan" -ref "$standard_template" -out "$output_scan" -omat "$output_directory/${scan_name}_registration.mat" >> mulreg_log.txt

        # You can also use fnirt for non-linear registration if needed
        # fnirt --in="$input_scan" --ref="$standard_template" --iout="$output_scan" --aff="$output_directory/${scan_name}_registration_affine.mat" --cout="$output_directory/${scan_name}_warp_coeff.nii.gz"
    done

    echo "Registration complete for subdirectory $subdirectory."
done

echo "All registrations complete."

chmod +x mulreg_script.sh
