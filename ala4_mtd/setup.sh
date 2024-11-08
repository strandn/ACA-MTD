cp ala4.pdb em.gro md.mdp topol.top phi2
cd phi2
gmx_mpi grompp -f md.mdp -c em.gro -p topol.top -o md_0_1.tpr
cd ..
cp ala4.pdb em.gro md.mdp topol.top psi2
cd psi2
gmx_mpi grompp -f md.mdp -c em.gro -p topol.top -o md_0_1.tpr
cd ..
cp ala4.pdb em.gro md.mdp topol.top phi3
cd phi3
gmx_mpi grompp -f md.mdp -c em.gro -p topol.top -o md_0_1.tpr
cd ..
cp ala4.pdb em.gro md.mdp topol.top psi3
cd psi3
gmx_mpi grompp -f md.mdp -c em.gro -p topol.top -o md_0_1.tpr
cd ..
cp ala4.pdb em.gro md.mdp topol.top phi4
cd phi4
gmx_mpi grompp -f md.mdp -c em.gro -p topol.top -o md_0_1.tpr
cd ..
cp ala4.pdb em.gro md.mdp topol.top psi4
cd psi4
gmx_mpi grompp -f md.mdp -c em.gro -p topol.top -o md_0_1.tpr
cd ..
