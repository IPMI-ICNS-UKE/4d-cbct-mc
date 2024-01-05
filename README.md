# CBCT_Simulation
## Commands
### To run the simulation
```
run-mc --data-folder /mnt/nas_io/anarchy/4d_cbct_mc/4d_ct_lung_uke_artifact_free --output-folder /mnt/nas_io/anarchy/4d_cbct_mc/speedup --reference --speedups 2 --speedups 5 --speedups 10 --phases 0 --segmenter-weights /mnt/nas_io/anarchy/4d_cbct_mc/segmenter/2023-09-21T17:18:03.218908_run_39a7956b4719411f99ddf071__step_95000.pth --segmenter-patch-overlap 0.75 --reconstruct --gpu 0 --gpu 1
```


run-mc --data-folder /datalake_fast/4d_ct_lung_uke_artifact_free --output-folder /datalake_fast/mc_output/3d --phases 0 --gpu 0 --reference --regex 024.* --n-projections 90

run-mc --data-folder /datalake_fast/4d_ct_lung_uke_artifact_free --output-folder /datalake_fast/mc_output/3d --phases 0 --gpu 0 --reference --regex 024.* --segmenter-weights /mnt/nas_io/anarchy/4d_cbct_mc/segmenter/2023-09-21T17:18:03.218908_run_39a7956b4719411f99ddf071__step_95000.pth --segmenter-patch-overlap 0.25 --segmenter-patch-shape 128 128 128

run-mc --data-folder /datalake_fast/4d_ct_lung_uke_artifact_free --output-folder /datalake_fast/mc_output/4d --phases 0  --gpu 0 --reference --regex 024.* --correspondence-model /mnt/nas_io/anarchy/4d_cbct_mc/024_correspondence_model.pkl --respiratory-signal /mnt/nas_io/anarchy/4d_cbct_mc/024_respiratory_signal.pkl

run-mc --data-folder /datalake_fast/4d_ct_lung_uke_artifact_free --output-folder /datalake_fast/mc_output/4d --phases 0 --gpu 0 --reference --regex 024.* --segmenter-weights /mnt/nas_io/anarchy/4d_cbct_mc/segmenter/2023-09-21T17:18:03.218908_run_39a7956b4719411f99ddf071__step_95000.pth --segmenter-patch-overlap 0.25 --segmenter-patch-shape 496 496 32 --correspondence-model /mnt/nas_io/anarchy/4d_cbct_mc/024_correspondence_model.pkl --respiratory-signal /mnt/nas_io/anarchy/4d_cbct_mc/024_respiratory_signal.pkl
