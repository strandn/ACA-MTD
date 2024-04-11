#!/bin/bash
 
##SBATCH --job-name=gaussian_biased_1e6
##SBATCH --output=%x.out
##SBATCH --error=%x.err

#SBATCH --job-name=gaussian_knn
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
 
#SBATCH --time=2-0:00:00
 
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
 
#SBATCH --export=NONE

export I_MPI_HYDRA_TOPOLIB=ipl
export JULIA_NUM_THREADS=1

echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

# julia gaussian_cv.jl
# mpiexecjl -n 1 julia gaussian_biased.jl
# julia gaussian_mtd.jl 20
mpiexecjl -n 10 julia gaussian_knn.jl $RANK $K
