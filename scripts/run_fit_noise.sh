#!/bin/bash


SESSION="4d-cbct-mc-fit-noise"

SESSIONEXISTS=$(tmux list-sessions | grep $SESSION)

# base call
CALL='fit-noise --output-folder /mnt/nas_io/anarchy/4d_cbct_mc/fit_noise --n-runs 3 --loglevel debug'

# Only create tmux session if it doesn't already exist
if [ "$SESSIONEXISTS" = "" ]; then
  # start new session
  tmux new-session -d -s $SESSION
  tmux rename-window -t 0 'main'

  # create all windows and run script
  tmux new-window -t $SESSION:1 -n 'gpu 0'
  tmux send-keys -t 'gpu 0' 'conda activate cbctmc' C-m "${CALL} --gpu 0" C-m

fi

tmux attach-session -t $SESSION:0
