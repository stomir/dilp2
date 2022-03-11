#!/bin/bash
if [ -n $SLURM_STEP_GPUS ]; then
    $@ --cuda 0 #$SLURM_STEP_GPUS
else
    $@
fi