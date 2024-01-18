#!/bin/bash

run-mc-lp --output-folder /mnt/nas_io/anarchy/4d_cbct_mc/mc_mtf_final/lp_0.50gap --gpu 0 --gpu 2 --gpu 4 --gpu 5 --line-gap 0.50 --reference --speedups 10.0 --speedups 20.0 --speedups 30.0 --speedups 40.0 --speedups 50.0 --reconstruct
run-mc-lp --output-folder /mnt/nas_io/anarchy/4d_cbct_mc/mc_mtf_final/lp_0.75gap --gpu 0 --gpu 2 --gpu 4 --gpu 5 --line-gap 0.75 --reference --speedups 10.0 --speedups 20.0 --speedups 30.0 --speedups 40.0 --speedups 50.0 --reconstruct
run-mc-lp --output-folder /mnt/nas_io/anarchy/4d_cbct_mc/mc_mtf_final/lp_1.00gap --gpu 0 --gpu 2 --gpu 4 --gpu 5 --line-gap 1.00 --reference --speedups 10.0 --speedups 20.0 --speedups 30.0 --speedups 40.0 --speedups 50.0 --reconstruct
run-mc-lp --output-folder /mnt/nas_io/anarchy/4d_cbct_mc/mc_mtf_final/lp_1.25gap --gpu 0 --gpu 2 --gpu 4 --gpu 5 --line-gap 1.25 --reference --speedups 10.0 --speedups 20.0 --speedups 30.0 --speedups 40.0 --speedups 50.0 --reconstruct
run-mc-lp --output-folder /mnt/nas_io/anarchy/4d_cbct_mc/mc_mtf_final/lp_1.50gap --gpu 0 --gpu 2 --gpu 4 --gpu 5 --line-gap 1.50 --reference --speedups 10.0 --speedups 20.0 --speedups 30.0 --speedups 40.0 --speedups 50.0 --reconstruct
run-mc-lp --output-folder /mnt/nas_io/anarchy/4d_cbct_mc/mc_mtf_final/lp_1.75gap --gpu 0 --gpu 2 --gpu 4 --gpu 5 --line-gap 1.75 --reference --speedups 10.0 --speedups 20.0 --speedups 30.0 --speedups 40.0 --speedups 50.0 --reconstruct
run-mc-lp --output-folder /mnt/nas_io/anarchy/4d_cbct_mc/mc_mtf_final/lp_2.50gap --gpu 0 --gpu 2 --gpu 4 --gpu 5 --line-gap 2.50 --reference --speedups 10.0 --speedups 20.0 --speedups 30.0 --speedups 40.0 --speedups 50.0 --reconstruct
run-mc-lp --output-folder /mnt/nas_io/anarchy/4d_cbct_mc/mc_mtf_final/lp_4.00gap --gpu 0 --gpu 2 --gpu 4 --gpu 5 --line-gap 4.00 --reference --speedups 10.0 --speedups 20.0 --speedups 30.0 --speedups 40.0 --speedups 50.0 --reconstruct
