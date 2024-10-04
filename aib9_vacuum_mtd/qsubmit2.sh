#!/bin/bash
 
#SBATCH --job-name=aib9_sumhills
#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --time=2-0:00:00
 
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
 
#SBATCH --export=NONE

export PLUMED_NUM_THREADS=8

echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

./sumhills.sh
