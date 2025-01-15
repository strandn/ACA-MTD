#!/bin/bash
 
#SBATCH --job-name=ala4_vacuum
#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --time=2-0:00:00
 
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
 
#SBATCH --ntasks=80
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
##SBATCH --exclude=midway3-[0250-0251]
 
#SBATCH --export=NONE

echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

# rm -f *ff* */\#md_0_1.* */bck.*
rm -f *ff*
for i in `seq 0 79`;
do
    cp ala4.pdb em.gro md.mdp plumed.dat topol.top $i
    cd $i
    gmx_mpi grompp -f md.mdp -c em.gro -p topol.top -o md_0_1.tpr
    cd ..
done
mpirun -np 80 gmx_mpi mdrun -deffnm md_0_1 -plumed plumed.dat -multidir 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79
for i in `seq 0 79`;
do
    sed '/#!/d' $i/colvar.$i.dat > $i/colvar.$i.dat.0
done
python3 merge_colvars.py
plumed driver --plumed plumed2.dat --noatoms
plumed driver --plumed plumed4.dat --noatoms
plumed driver --plumed plumed3.dat --noatoms
