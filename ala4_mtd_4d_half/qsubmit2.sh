#!/bin/bash
 
#SBATCH --job-name=ala4_histo
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

echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

export PLUMED_NUM_THREADS=8

rm -f *ff*
plumed driver --plumed plumed2.dat --noatoms
sed "14,500013d" COLVAR > COLVAR.0
sed "14,1000013d" COLVAR > COLVAR.1
plumed driver --plumed plumed3.dat --noatoms
plumed driver --plumed plumed4.dat --noatoms
