#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
        arr=($line)
        sbatch --export=RC=${arr[0]},NBASIS=${arr[1]},NSAMPLES=${arr[2]} qsubmit.sh
done < "$1"
