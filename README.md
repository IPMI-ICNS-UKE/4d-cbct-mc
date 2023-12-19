# CBCT_Simulation
## Commands
### To run the simulation
```
run-mc --data-folder /mnt/nas_io/anarchy/4d_cbct_mc/4d_ct_lung_uke_artifact_free --output-folder /mnt/nas_io/anarchy/4d_cbct_mc/speedup --reference --speedups 2 --speedups 5 --speedups 10 --phases 0 --segmenter-weights /mnt/nas_io/anarchy/4d_cbct_mc/segmenter/2023-09-21T17:18:03.218908_run_39a7956b4719411f99ddf071__step_95000.pth --segmenter-patch-overlap 0.75 --reconstruct --gpu 0 --gpu 1
```
