#!/bin/bash

sed -i '$d' */colvar.*
for i in `seq 0 39`; do sed '/#!/d' $i/colvar.$i.dat > $i/colvar.$i.dat.0; done
python3 merge_colvars.py
