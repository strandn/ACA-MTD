#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
        arr=($line)
        sbatch --export=RANK=${arr[0]},K=${arr[1]} qsubmit.sh
done < "$1"
