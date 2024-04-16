#!/bin/bash
 
##SBATCH --job-name=gaussian_cv
##SBATCH --output=%x.out
##SBATCH --error=%x.err

#SBATCH --job-name=gaussian_cv
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
 
#SBATCH --time=2-0:00:00
 
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=100G
 
#SBATCH --export=NONE

export I_MPI_HYDRA_TOPOLIB=ipl
export JULIA_NUM_THREADS=48

echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

julia gaussian_cv.jl
# mpiexecjl -n 1 julia gaussian_biased.jl
# julia gaussian_mtd.jl 20
# mpiexecjl -n 10 julia gaussian_knn.jl $RANK $K
