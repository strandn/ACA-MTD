#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
        arr=($line)
        sbatch --export=R=${arr[0]},RC=${arr[1]},BASIS=${arr[2]} qsubmit.sh
done < "$1"
