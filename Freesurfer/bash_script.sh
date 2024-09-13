#!/bin/bash

module load freesurfer

# use srun to run this script!!!
# srun -p sapphire --pty --nodes=1 --cpus-per-task=64 --time=12:00:00 bash
# check here for flags https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all

#set SUBJECTS_DIR
export SUBJECTS_DIR=/home/cmb247/Desktop/Project_3/Freesurfer/
echo "SUBJECTS_DIR is set to: $SUBJECTS_DIR"

#Uncomment each step below:
#recon-all -s 19978_ultra-fast -autorecon1 -notalairach -noskullstrip
#recon-all -s 19978_ultra-fast -autorecon2-cp -noaseg -nofix
#recon-all -s 19978_ultra-fast -autorecon2 -noaseg -nofix
recon-all -s 19978_ultra-fast -autorecon3 -nofix
