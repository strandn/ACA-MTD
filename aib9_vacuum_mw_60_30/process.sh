#!/bin/bash

for i in `seq 0 39`;
do
    sed -i '$d' $i/colvar.$i.dat
    sed '/#!/d' $i/colvar.$i.dat > $i/colvar.$i.dat.0
done
python3 merge_colvars.py
