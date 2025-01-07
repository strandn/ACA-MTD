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
sed -i '$d' */colvar.*
for i in `seq 0 39`; do sed '/#!/d' $i/colvar.$i.dat > $i/colvar.$i.dat.0; done
python3 merge_colvars.py
plumed driver --plumed plumed2.dat --noatoms
sed "14,1250053d" colvar.dat > colvar_half.dat
plumed driver --plumed plumed3.dat --noatoms
