#!/bin/bash
 
#SBATCH --job-name=ala3_mtd
#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --time=2-0:00:00
 
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
 
#SBATCH --export=NONE

export OMP_NUM_THREADS=6

echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

mpirun -np 4 gmx_mpi mdrun -deffnm md_0_1 -plumed plumed.dat -multidir phi2 psi2 phi3 psi3 -replex 2000
