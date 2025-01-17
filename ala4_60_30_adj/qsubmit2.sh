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
sed "14,1000013d" colvar.dat > colvar_half.dat
plumed driver --plumed plumed2.dat --noatoms
plumed driver --plumed plumed3.dat --noatoms
plumed driver --plumed plumed4.dat --noatoms
plumed driver --plumed plumed5.dat --noatoms
plumed driver --plumed plumed6.dat --noatoms
