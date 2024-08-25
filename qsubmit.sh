#!/bin/bash
 
#SBATCH --job-name=gaussian_sketch
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
 
#SBATCH --time=2-0:00:00
 
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
 
#SBATCH --export=NONE

export JULIA_NUM_THREADS=8

echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

julia gaussian_2d.jl $RC $NBASIS $NSAMPLES
#julia gaussian_4d.jl $RC $NBASIS $NSAMPLES
