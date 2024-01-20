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


run-mc --data-folder /mnt/nas_io/anarchy/4d_cbct_mc/4d_ct_lung_uke_artifact_free --output-folder /mnt/nas_io/anarchy/4d_cbct_mc/speedup --phases 0 --gpu 0 --gpu 1 --gpu 2 --gpu 4 --gpu 5  --speedups 20 --speedups 50 --regex 024.* --forward-projection --reconstruct




run-mc --data-folder /data_l79a/fmadesta/4d_cbct/R4DCIRS/4DCT --output-folder /datalake_fast/mc_output/4d_cirs --phases 2 --gpu 0 --speedups 10 --regex phase_based_gt_curve --segmenter-weights /mnt/nas_io/anarchy/4d_cbct_mc/segmenter/2023-09-21T17:18:03.218908_run_39a7956b4719411f99ddf071__step_95000.pth --segmenter-patch-overlap 0.25 --segmenter-patch-shape 288 288 32 --correspondence-model /mnt/nas_io/anarchy/4d_cbct_mc/cirs_correspondence_model.pkl --respiratory-signal /mnt/nas_io/anarchy/4d_cbct_mc/cirs_varian_respiratory_signal.pkl --respiratory-signal-quantization 20 --cirs-phantom --reconstruct

run-mc --output-folder /datalake_fast/mc_output/4d_cirs --phases 2 --gpu 1 --reference --geometry-filepath /data_l79a/fmadesta/4d_cbct/R4DCIRS/for_mc/4d_cirs/cirs_phase_02.pkl --correspondence-model /data_l79a/fmadesta/4d_cbct/R4DCIRS/for_mc/4d_cirs/cirs_correspondence_model.pkl --respiratory-signal /data_l79a/fmadesta/4d_cbct/R4DCIRS/for_mc/4d_cirs/cirs_varian_respiratory_signal.pkl --respiratory-signal-quantization 40 --cirs-phantom --reconstruct





CatPhan runs:
run-mc --output-folder /mnt/nas_io/anarchy/4d_cbct_mc/catphan --gpu 0 --gpu 1 --gpu 2 --reference --speedups 10.0 --speedups 20.0 --speedups 50.0 --catphan-phantom --reconstruct


4D CIRS runs:
run-mc --output-folder /datalake_fast/mc_output/4d_cirs --phases 2 --gpu 1 --gpu 0 --reference --geometry-filepath /data_l79a/fmadesta/4d_cbct/R4DCIRS/for_mc/4d_cirs/cirs_phase_02.pkl --correspondence-model /data_l79a/fmadesta/4d_cbct/R4DCIRS/for_mc/4d_cirs/cirs_correspondence_model.pkl --respiratory-signal /data_l79a/fmadesta/4d_cbct/R4DCIRS/for_mc/4d_cirs/cirs_varian_respiratory_signal.pkl --respiratory-signal-quantization 20 --cirs-phantom --reconstruct --forward-projection
run-mc --output-folder /datalake_fast/mc_output/4d_cirs --phases 2 --gpu 1 --gpu 0 --speedups 20 --geometry-filepath /data_l79a/fmadesta/4d_cbct/R4DCIRS/for_mc/4d_cirs_large/cirs_phase_02.pkl --correspondence-model /data_l79a/fmadesta/4d_cbct/R4DCIRS/for_mc/4d_cirs_large/cirs_correspondence_model.pkl --respiratory-signal /data_l79a/fmadesta/4d_cbct/R4DCIRS/for_mc/4d_cirs_large/cirs_varian_respiratory_signal.pkl --respiratory-signal-quantization 20 --cirs-phantom --reconstruct --forward-projection

4D patient runs:
run-mc --data-folder /mnt/nas_io/anarchy/4d_cbct_mc/4d_ct_lung_uke_artifact_free --output-folder /mnt/nas_io/anarchy/4d_cbct_mc/4d --phases 2  --gpu 0 --gpu 1 --gpu 2 --reference --speedups 10 --speedups 20 --regex 024.* --correspondence-model /mnt/nas_io/anarchy/4d_cbct_mc/024_correspondence_model_nonmasked.pkl --respiratory-signal /mnt/nas_io/anarchy/4d_cbct_mc/024_respiratory_signal.pkl --segmenter-weights /mnt/nas_io/anarchy/4d_cbct_mc/segmenter/2023-09-21T17:18:03.218908_run_39a7956b4719411f99ddf071__step_95000.pth --reconstruct --forward-projection --respiratory-signal-quantization 20
