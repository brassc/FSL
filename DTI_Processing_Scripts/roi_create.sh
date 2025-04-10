#!/bin/bash
# Input parameters
patient_id=$1
timepoint=$2
dwi_space_path=$3 # Path to the diffusion weighted image

# Get coordinates from CSV using the patient_id and timepoint
# This would require a way to look up in your CSV - could use awk/grep
ant_x=$(grep "$patient_id,$timepoint" coordinates.csv | awk -F, '{print $12}')
ant_y=$(grep "$patient_id,$timepoint" coordinates.csv | awk -F, '{print $13}')
ant_z=$(grep "$patient_id,$timepoint" coordinates.csv | awk -F, '{print $14}')
post_x=$(grep "$patient_id,$timepoint" coordinates.csv | awk -F, '{print $15}')
post_y=$(grep "$patient_id,$timepoint" coordinates.csv | awk -F, '{print $16}')
post_z=$(grep "$patient_id,$timepoint" coordinates.csv | awk -F, '{print $17}')
baseline_ant_x=$(grep "$patient_id,$timepoint" coordinates.csv | awk -F, '{print $18}')
baseline_ant_y=$(grep "$patient_id,$timepoint" coordinates.csv | awk -F, '{print $19}')
baseline_ant_z=$(grep "$patient_id,$timepoint" coordinates.csv | awk -F, '{print $20}')
baseline_post_x=$(grep "$patient_id,$timepoint" coordinates.csv | awk -F, '{print $21}')
baseline_post_y=$(grep "$patient_id,$timepoint" coordinates.csv | awk -F, '{print $22}')
baseline_post_z=$(grep "$patient_id,$timepoint" coordinates.csv | awk -F, '{print $23}')

# Create spherical ROIs for each point
output_dir="output/${patient_id}/${timepoint}"
mkdir -p $output_dir

# Create anterior ROI
fslmaths $dwi_space_path -mul 0 -add 1 -roi $ant_x 1 $ant_y 1 $ant_z 1 0 1 $output_dir/ant_point -odt float

# Create posterior ROI
fslmaths $dwi_space_path -mul 0 -add 1 -roi $post_x 1 $post_y 1 $post_z 1 0 1 $output_dir/post_point -odt float

# Create baseline anterior ROI
fslmaths $dwi_space_path -mul 0 -add 1 -roi $baseline_ant_x 1 $baseline_ant_y 1 $baseline_ant_z 1 0 1 $output_dir/baseline_ant_point -odt float

# Create baseline posterior ROI
fslmaths $dwi_space_path -mul 0 -add 1 -roi $baseline_post_x 1 $baseline_post_y 1 $baseline_post_z 1 0 1 $output_dir/baseline_post_point -odt float

# Create rings around anterior point
for i in {1..5}; do
    radius=$i
    prev_radius=$((i-1))
    if [ $prev_radius -eq 0 ]; then
        # First ring
        fslmaths $output_dir/ant_point -kernel sphere $radius -dilM $output_dir/ant_ring${i}
    else
        # Subsequent rings (donuts)
        fslmaths $output_dir/ant_point -kernel sphere $radius -dilM $output_dir/ant_whole_${i}
        fslmaths $output_dir/ant_whole_${i} -sub $output_dir/ant_whole_$((i-1)) $output_dir/ant_ring${i}
    fi
done

# Create rings around posterior point
for i in {1..5}; do
    radius=$i
    prev_radius=$((i-1))
    if [ $prev_radius -eq 0 ]; then
        # First ring
        fslmaths $output_dir/post_point -kernel sphere $radius -dilM $output_dir/post_ring${i}
    else
        # Subsequent rings (donuts)
        fslmaths $output_dir/post_point -kernel sphere $radius -dilM $output_dir/post_whole_${i}
        fslmaths $output_dir/post_whole_${i} -sub $output_dir/post_whole_$((i-1)) $output_dir/post_ring${i}
    fi
done

# Create rings around baseline anterior point
for i in {1..5}; do
    radius=$i
    prev_radius=$((i-1))
    if [ $prev_radius -eq 0 ]; then
        # First ring
        fslmaths $output_dir/baseline_ant_point -kernel sphere $radius -dilM $output_dir/baseline_ant_ring${i}
    else
        # Subsequent rings (donuts)
        fslmaths $output_dir/baseline_ant_point -kernel sphere $radius -dilM $output_dir/baseline_ant_whole_${i}
        fslmaths $output_dir/baseline_ant_whole_${i} -sub $output_dir/baseline_ant_whole_$((i-1)) $output_dir/baseline_ant_ring${i}
    fi
done

# Create rings around baseline posterior point
for i in {1..5}; do
    radius=$i
    prev_radius=$((i-1))
    if [ $prev_radius -eq 0 ]; then
        # First ring
        fslmaths $output_dir/baseline_post_point -kernel sphere $radius -dilM $output_dir/baseline_post_ring${i}
    else
        # Subsequent rings (donuts)
        fslmaths $output_dir/baseline_post_point -kernel sphere $radius -dilM $output_dir/baseline_post_whole_${i}
        fslmaths $output_dir/baseline_post_whole_${i} -sub $output_dir/baseline_post_whole_$((i-1)) $output_dir/baseline_post_ring${i}
    fi
done

# Print confirmation message
echo "ROIs created successfully for patient $patient_id at timepoint $timepoint"
echo "Output stored in $output_dir"