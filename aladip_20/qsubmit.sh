#!/bin/bash
 
#SBATCH --job-name=aladip_vacuum
#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --time=2-0:00:00
 
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
 
#SBATCH --export=NONE

echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

source ~/.bashrc

rm -f analysis.* */\#md_0_1.* */bck.*
for i in `seq 0 9`;
do
    cp diala.pdb em.gro md.mdp plumed.dat topol.top $i
    cd $i
    gmx_mpi grompp -f md.mdp -c em.gro -p topol.top -o md_0_1.tpr
    cd ..
done
mpirun -np 10 gmx_mpi mdrun -deffnm md_0_1 -plumed plumed.dat -multidir 0 1 2 3 4 5 6 7 8 9
# mpirun -np 10 gmx_mpi mdrun -deffnm md_0_1 -multidir 0 1 2 3 4 5 6 7 8 9
# module load valgrind
# mpirun -np 10 valgrind --leak-check=full --track-origins=yes gmx_mpi mdrun -deffnm md_0_1 -multidir 0 1 2 3 4 5 6 7 8 9
# mpirun -np 2 gmx_mpi mdrun -deffnm md_0_1 -plumed plumed.dat -multidir 0 1
# gmx_mpi grompp -f md.mdp -c em.gro -p topol.top -o md_0_1.tpr
# gmx_mpi mdrun -deffnm md_0_1 -plumed plumed.dat
# for i in `seq 0 9`;
# do
#     sed '/#!/d' $i/colvar.$i.dat > $i/colvar.$i.dat.0
# done
# python3 merge_colvars.py
# plumed driver --plumed plumed2.dat --noatoms
