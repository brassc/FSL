#!/bin/bash
module load fsl

#RUN FROM "BET_Extractions" DIRECTORY LEVEL
#THIS SCRIPT DYNAMICALLY SELECTS THE EARLIEST T1 SCAN (TIME 1) IN THE 'OG_Scans' DIRECTORY AND REGISTERS ALL SUBSEQUENT T1 TO THAT 'EARLIEST' SCAN. 

registration_space_name="T1w_time1"

# Define the priority of keywords in an array
keywords=("ultra-fast" "fast" "acute" "3mo" "6mo" "12mo" "24mo")

# List of subdirectory names (e.g., 12519, 13990, 14324, 16577, 20942)
subdirectories=("12519" "13198" "13782" "13990" "14324" "16754" "19344" "19575" "19978" "19981" "20174" "20651" "20942" "21221" "22725" "22785" "23348" ) #

# Loop through each subdirectory
for subdirectory in "${subdirectories[@]}"; do
    input_directory="${subdirectory}/OG_Scans"
    output_directory="${subdirectory}/${registration_space_name}_registered_scans"

    # Initialize variable to hold the name of the earliest file
    earliest_file=""

    # Loop through each keyword in order of priority
    for keyword in "${keywords[@]}"; do
      # Search files containing the current keyword in the input directory
      files=$(find "$input_directory" -type f -name "T1w_*$keyword*.nii.gz")
  
      # If files are found for the current keyword
      if [ -n "$files" ]; then
        # Select the first file as the earliest (assuming any file matches the criteria)
        for file in $files; do
          earliest_file=$file
          echo "$earliest_file"
          break 2 # Exit both loops since the earliest file is found
        done
      fi
    done
     
    # Define the T1 template path 
    standard_template=$earliest_file
	echo "$standard_template"
    # Make sure the output directory exists; if not, create it
    mkdir -p "$output_directory"

    # Loop through the brain scans in the input directory
    for input_scan in "$input_directory"/*.nii.gz; do
        # Extract the scan filename without the path and extension
        scan_name=$(basename "${input_scan%.nii.gz}")

        # Define the output path for the registered scan
        output_scan="$output_directory/$registration_space_name.${scan_name}_registered.nii.gz"

        # Perform registration using flirt or fnirt (adjust parameters as needed)
	echo "Completing registration for $input_scan to $standard_template" >> mulreg_log.txt
        flirt -in "$input_scan" -ref "$standard_template" -out "$output_scan" -omat "$output_directory/${scan_name}_registration.mat" >> mulreg_log.txt

        # You can also use fnirt for non-linear registration if needed
        # fnirt --in="$input_scan" --ref="$standard_template" --iout="$output_scan" --aff="$output_directory/${scan_name}_registration_affine.mat" --cout="$output_directory/${scan_name}_warp_coeff.nii.gz"
    done

    echo "Registration complete for subdirectory $subdirectory."
done

echo "All registrations complete."

chmod +x t1mulreg_script.sh
