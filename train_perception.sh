#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=5
ulimit -Sn 600000
python chauffeur.py train \
    -e mm45_v4_PcSensordropLessmap_rfsv4_extra_structure_noise_lanecolor \
    -m 0.05
