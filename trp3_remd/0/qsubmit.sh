#!/bin/bash
 
#SBATCH --job-name=trp3_histo
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

export PLUMED_NUM_THREADS=8

rm -f analysis.* bck.*
plumed driver --plumed plumed2.dat --noatoms
