#!/bin/bash
 
#SBATCH --job-name=aib9_vacuum_mtd
#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --time=2-0:00:00
 
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
 
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=9
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
 
#SBATCH --export=NONE

export OMP_NUM_THREADS=4

echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

mpirun -np 18 gmx_mpi mdrun -deffnm md_0_1 -plumed plumed.dat -multidir phi1 psi1 phi2 psi2 phi3 psi3 phi4 psi4 phi5 psi5 phi6 psi6 phi7 psi7 phi8 psi8 phi9 psi9 -replex 2000
