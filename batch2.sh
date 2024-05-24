#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
        arr=($line)
        sbatch --export=BF=${arr[0]},W=${arr[1]} qsubmit.sh
done < "$1"
