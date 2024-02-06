#!/bin/bash

#RUN FROM "BET_Extractions" DIRECTORY LEVEL

module load fsl

# Define the standard space template path (change this to your standard space template)
standard_template="mni_icbm152_t1_tal_nlin_sym_55_ext.nii"
standard_space_name="mni152_fnirt"


# List of subdirectory names (e.g., 12519, 13990, 14324, 16577, 20942)
subdirectories=("19978" "19981" "20174" "20651" "20942" "21221" "22725" "22785" "23348") # "12519" "13198" "13782" "13990" "14324" "16754" "19344" "19575" 
# Define the priority of keywords in an array
keywords=("ultra-fast" "fast" "acute" "3mo" "6mo" "12mo" "24mo")


# Loop through each subdirectory
for subdirectory in "${subdirectories[@]}"; do
    input_directory="${subdirectory}/OG_Scans"
    output_directory="${subdirectory}/MNI152_fnirt_registered_scans"

    # Make sure the output directory exists; if not, create it
    mkdir -p "$output_directory"

    # Loop through the brain scans in the input directory
    for input_scan in "$input_directory"/*.nii.gz; do
        # Extract the scan filename without the path and extension
        scan_name=$(basename "${input_scan%.nii.gz}")

        # Define the output path for the registered scan
        output_scan="$output_directory/${standard_space_name}_${scan_name}_registered.nii.gz"
        
        # Iterate through the list of keywords
        for keyword in "${keywords[@]}"; do
          if [[ $scan_name == *"$keyword"* ]]; then
            extracted_keyword="${filename%%$keyword*}$keyword"
            echo "Extracted keyword: $extracted_keyword"
          fi
        done

        # Perform registration using flirt or fnirt (adjust parameters as needed)
	echo "Completing linear registration for $patientID $extracted_keyword scan" >> mulfnirtreg_log.txt
        flirt -in "$input_scan" -ref "$standard_template" -out "$output_scan" -omat "$output_directory/${scan_name}_registration.mat" >> mulfnirtreg_log.txt
        echo "Linear registration for $patientID $extracted_keyword complete. Starting non-linear registration" >> mulfnirtreg_log.txt
        # You can also use fnirt for non-linear registration if needed
        fnirt --in="$input_scan" --ref="$standard_template" --iout="$output_scan" --aff="$output_directory/${scan_name}_registration.mat" --cout="$output_directory/${scan_name}_warp_coeff.nii.gz" >> mulfnirtreg_log.txt
        echo "Non-linear registration for $patientID $extracted_keyword complete." >> mulfnirtreg_log.txt
    done

    echo "Registration complete for subdirectory $subdirectory."
done

echo "All registrations complete."

chmod +x mulfnirtreg_script.sh
