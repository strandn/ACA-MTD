#!/bin/bash
 
#SBATCH --job-name=ala3_remd
#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --time=2-0:00:00
 
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
 
#SBATCH --export=NONE

echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

mpirun -np 8 gmx_mpi mdrun -deffnm md_0_1 -plumed plumed.dat -multidir 0 1 2 3 4 5 6 7 -replex 2000
