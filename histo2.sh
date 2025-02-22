#!/bin/bash

cd ala3_30_30
rm -f analysis.*
plumed driver --plumed plumed2.dat --noatoms
cd ..
cd ala3_40_20_10_wt
rm -f analysis.*
plumed driver --plumed plumed2.dat --noatoms
cd ..
cd ala3_40_40_10
rm -f analysis.*
plumed driver --plumed plumed2.dat --noatoms
cd ..
cd ala3_40_40_wt
rm -f analysis.*
plumed driver --plumed plumed2.dat --noatoms
cd ..
cd ala3_60_30
rm -f analysis.*
plumed driver --plumed plumed2.dat --noatoms
cd ..
cd ala3_60_30_wt
rm -f analysis.*
plumed driver --plumed plumed2.dat --noatoms
cd ..
