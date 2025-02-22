#!/bin/bash
 
#SBATCH --job-name=ala4_vacuum_3d
#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --time=2-0:00:00
 
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
 
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --exclude=midway3-[0250-0251]
 
#SBATCH --export=NONE

echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

rm -f *ff* */\#md_0_1.* */bck.* bck.*
for i in `seq 0 19`;
do
    cp ala4.pdb em.gro md.mdp plumed.dat topol.top $i
    cd $i
    gmx_mpi grompp -f md.mdp -c em.gro -p topol.top -o md_0_1.tpr
    cd ..
done
mpirun -np 20 gmx_mpi mdrun -deffnm md_0_1 -plumed plumed.dat -multidir 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
for i in `seq 0 19`;
do
    sed '/#!/d' $i/colvar.$i.dat > $i/colvar.$i.dat.0
done
python3 merge_colvars.py
plumed driver --plumed plumed2.dat --noatoms
sed "14,1000013d" colvar.dat > colvar.dat.0
sed "14,2500013d" colvar.dat > colvar.dat.1
plumed driver --plumed plumed3.dat --noatoms
plumed driver --plumed plumed4.dat --noatoms
