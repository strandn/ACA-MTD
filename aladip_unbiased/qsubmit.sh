#!/bin/bash
 
#SBATCH --job-name=aladip_vacuum_unbiased
#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --time=2-0:00:00
 
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
 
#SBATCH --export=NONE

echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

gmx_mpi mdrun -deffnm md_0_1 -plumed plumed.dat
