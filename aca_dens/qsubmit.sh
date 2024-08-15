#!/bin/bash
 
#SBATCH --job-name=gaussian_acadens
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
 
#SBATCH --time=2-0:00:00
 
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
 
#SBATCH --export=NONE

export JULIA_NUM_THREADS=4

echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

julia gaussian_2d.jl $RC $NBASIS $NSAMPLES
