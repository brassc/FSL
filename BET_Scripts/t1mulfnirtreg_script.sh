#!/bin/bash
module load fsl

#RUN FROM "BET_Extractions" DIRECTORY LEVEL
#THIS SCRIPT DYNAMICALLY SELECTS THE EARLIEST T1 SCAN (TIME 1) IN THE 'T1w_time1_registered_scans' DIRECTORY AND REGISTERS ALL SUBSEQUENT T1 TO THAT 'EARLIEST' SCAN. 


registration_space_name="T1w_time1_fnirt"

# Define the priority of keywords in an array
keywords=("ultra-fast" "fast" "acute" "3mo" "6mo" "12mo" "24mo")

# List of subdirectory names (e.g., 12519, 13990, 14324, 16577, 20942)
subdirectories=("12519" "13198" "13782" "13990" "14324" "16754" "19344" "19575" "19978" "19981" "20174" "20651" "20942" "21221" "22725" "22785" "23348" ) #

log_file="mult1fnirtreg_log.txt"

# Loop through each subdirectory
for subdirectory in "${subdirectories[@]}"; do
    flirt_directory="${subdirectory}/T1w_time1_registered_scans"
    input_directory="${subdirectory}/OG_Scans"
    output_directory="${subdirectory}/${registration_space_name}_registered_scans"
    patientID=$subdirectory

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
          #get log readout
          for keyword in "${keywords[@]}"; do
            if [[ $earliest_file == *"$keyword"* ]]; then
              earliest_keyword=$keyword
              echo "Earliest scan for $patientID: $earliest_keyword">>$log_file
              break # Stop the loop once a match is found
            fi
          done
          #echo "$earliest_file">> $log_file
          break 2 # Exit both loops since the earliest file is found
        done
      fi
    done
     
    # Define the T1 template path 
    standard_template=$earliest_file

    # Make sure the output directory exists; if not, create it
    mkdir -p "$output_directory"

    # Loop through the brain scans in the input directory
    for input_scan in "$input_directory"/*.nii.gz; do
        # Extract the scan filename without the path and extension
        scan_name=$(basename "${input_scan%.nii.gz}")
        ##get --aff .mat filename i.e. scan_name w/o 'registered'        
        #affine_mat_name=${scan_name:0:-11} 
         
        for keyword in "${keywords[@]}"; do
          if [[ $scan_name == *"$keyword"* ]]; then
            echo "Matched keyword: $keyword">>$log_file
            break # Stop the loop once a match is found
          fi
        done

        # Define the output path for the registered scan
        output_scan="$output_directory/${registration_space_name}_${scan_name}_fnirt_registered.nii.gz"

        # Perform registration using flirt or fnirt (adjust parameters as needed)
         
        if [ "$earliest_keyword" != "$keyword" ]; then
	  echo "Completing non-linear registration for $patientID $keyword to $patientID $earliest_keyword" >> $log_file
          #echo "--aff=$input_directory/${affine_mat_name}_registration.mat"
          fnirt --in="$input_scan" --ref="$standard_template" --iout="$output_scan" --aff="$flirt_directory/${scan_name}_registration.mat" --cout="$output_directory/${scan_name}_fnirt_warp_coeff.nii.gz" >> $patientID/t1mulfnirtcmds_log.txt
        else
          : #do nothing
        fi
        #flirt -in "$input_scan" -ref "$standard_template" -out "$output_scan" -omat "$output_directory/${scan_name}_registration.mat" >> $log_file

        
    done

    echo "Registration complete for subdirectory $subdirectory." >> $log_file
done

echo "All registrations complete."

chmod +x t1mulfnirtreg_script.sh
