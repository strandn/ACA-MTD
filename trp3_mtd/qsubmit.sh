#!/bin/bash
 
#SBATCH --job-name=trp3_mtd
#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --time=2-0:00:00
 
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
 
#SBATCH --export=NONE

export OMP_NUM_THREADS=4

echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

./setup.sh
mpirun -np 8 gmx_mpi mdrun -deffnm md_0_1 -plumed plumed.dat -multidir phi1 psi1 chi11 chi12 phi2 psi2 chi21 chi22 -replex 2000
./sumhills
