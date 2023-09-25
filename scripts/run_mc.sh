#!/bin/bash


SESSION="4d-cbct-mc"

SESSIONEXISTS=$(tmux list-sessions | grep $SESSION)

# base call
CALL='run-mc --data-folder /mnt/nas_io/anarchy/4d_cbct_mc/4d_ct_lung_uke_artifact_free --regex ".*" --output-folder /mnt/nas_io/anarchy/4d_cbct_mc/output --segmenter-patch-overlap 0.75 --segmenter-patch-shape 384 384 64 --reference --speedups 10 --speedups 5 --speedups 2 --phases 0 --segmenter-weights /mnt/nas_io/anarchy/4d_cbct_mc/segmenter/2023-09-24T23\:17\:10.885863_run_6f02859e05e74f778a207756__step_15000.pth --reconstruct --loglevel debug'

# Only create tmux session if it doesn't already exist
if [ "$SESSIONEXISTS" = "" ]; then
  # start new session
  tmux new-session -d -s $SESSION
  tmux rename-window -t 0 'main'

  # create all windows and run script
  tmux new-window -t $SESSION:1 -n 'worker 1/gpu 0'
  tmux send-keys -t 'worker 1' 'conda activate cbctmc' C-m "${CALL} --n-workers 5 --i-worker 1 --gpu 0" C-m

  tmux new-window -t $SESSION:2 -n 'worker 2/gpu 1'
  tmux send-keys -t 'worker 2' 'conda activate cbctmc' C-m "${CALL} --n-workers 5 --i-worker 2 --gpu 1" C-m

  tmux new-window -t $SESSION:3 -n 'worker 3/gpu 2'
  tmux send-keys -t 'worker 3' 'conda activate cbctmc' C-m "${CALL} --n-workers 5 --i-worker 3 --gpu 2" C-m

  tmux new-window -t $SESSION:4 -n 'worker 4/gpu 4'
  tmux send-keys -t 'worker 4' 'conda activate cbctmc' C-m "${CALL} --n-workers 5 --i-worker 4 --gpu 4" C-m

  tmux new-window -t $SESSION:5 -n 'worker 5/gpu 5'
  tmux send-keys -t 'worker 5' 'conda activate cbctmc' C-m "${CALL} --n-workers 5 --i-worker 5 --gpu 5" C-m
fi

tmux attach-session -t $SESSION:0
