#!/bin/bash
 
#SBATCH --job-name=gaussian_50_7
#SBATCH --output=%x.out
#SBATCH --error=%x.err
 
#SBATCH --time=2:00:00
 
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
 
#SBATCH --export=NONE

export I_MPI_HYDRA_TOPOLIB=ipl
export JULIA_NUM_THREADS=1

echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

# julia gaussian_cv.jl
julia gaussian_biased.jl
