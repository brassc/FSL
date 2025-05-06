#!/bin/bash
#!
#! SLURM job script for Abaqus on Peta4-CascadeLake
#!

#! Name of the job:
#SBATCH -J roi_extraction
#! Which project should be charged:
#SBATCH -A MENON-SL3-CPU
#SBATCH -p cclake
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#SBATCH --ntasks=1
#! How many CPUs per task (for threaded applications)
#SBATCH --cpus-per-task=1
#! How much wallclock time will be required?
#SBATCH --time=8:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=BEGIN,END,FAIL

#! Load required modules
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-ccl
module load fsl

#! Set OpenMP threads for the threaded job
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#! Work directory
workdir="$SLURM_SUBMIT_DIR"
cd $workdir

# Command to run Abaqus with hybrid parallelization
CMD="./DTI_Processing_Scripts/roi_main.sh --num_bins=5 --bin_size=4"
CMD2="./DTI_Processing_Scripts/roi_main.sh --num_bins=10 --bin_size=2"


echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD
echo -e "\nExecuting command:\n==================\n$CMD2\n"
eval $CMD2
