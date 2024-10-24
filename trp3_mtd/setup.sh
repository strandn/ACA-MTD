cp trp3.pdb em.gro md.mdp topol.top phi1
cd phi1
gmx_mpi grompp -f md.mdp -c em.gro -p topol.top -o md_0_1.tpr
cd ..
cp trp3.pdb em.gro md.mdp topol.top psi1
cd psi1
gmx_mpi grompp -f md.mdp -c em.gro -p topol.top -o md_0_1.tpr
cd ..
cp trp3.pdb em.gro md.mdp topol.top chi11
cd chi11
gmx_mpi grompp -f md.mdp -c em.gro -p topol.top -o md_0_1.tpr
cd ..
cp trp3.pdb em.gro md.mdp topol.top chi12
cd chi12
gmx_mpi grompp -f md.mdp -c em.gro -p topol.top -o md_0_1.tpr
cd ..
cp trp3.pdb em.gro md.mdp topol.top phi2
cd phi2
gmx_mpi grompp -f md.mdp -c em.gro -p topol.top -o md_0_1.tpr
cd ..
cp trp3.pdb em.gro md.mdp topol.top psi2
cd psi2
gmx_mpi grompp -f md.mdp -c em.gro -p topol.top -o md_0_1.tpr
cd ..
cp trp3.pdb em.gro md.mdp topol.top chi21
cd chi21
gmx_mpi grompp -f md.mdp -c em.gro -p topol.top -o md_0_1.tpr
cd ..
cp trp3.pdb em.gro md.mdp topol.top chi22
cd chi22
gmx_mpi grompp -f md.mdp -c em.gro -p topol.top -o md_0_1.tpr
cd ..
