#!/bin/bash
 
#SBATCH --job-name=ala3_histo
#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --time=2-0:00:00
 
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
 
#SBATCH --export=NONE

export PLUMED_NUM_THREADS=8

echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

for i in `seq 0 19`;
do
    sed '/#!/d' $i/colvar.$i.dat > $i/colvar.$i.dat.0
done
python3 merge_colvars.py
plumed driver --plumed plumed2.dat --noatoms
sed "10,250009d" colvar.dat > colvar.dat.0
plumed driver --plumed plumed3.dat --noatoms
