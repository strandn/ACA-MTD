#!/bin/bash
 
#SBATCH --job-name=ala3_vacuum_4d
#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --time=2-0:00:00
 
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
 
#SBATCH --export=NONE

echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

rm -f analysis.* \#md_0_1.* bck.*
gmx_mpi mdrun -deffnm md_0_1 -plumed plumed.dat
plumed driver --plumed plumed2.dat --noatoms
