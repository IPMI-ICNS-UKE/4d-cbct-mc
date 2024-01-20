#!/bin/bash

for ((i=114; i<=1000; i++))
do
    output_folder="/mnt/nas_io/anarchy/4d_cbct_mc/catphan_variance_runs/catphan_variance_run_${i}"
    run-mc --output-folder "$output_folder" --gpu 0 --reference --catphan-phantom --n-projections 8 --random-seed $i
done
