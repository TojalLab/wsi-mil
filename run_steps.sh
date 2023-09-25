#!/bin/bash
export PL_DISABLE_FORK=1

CONFIG=configs/example_config.yaml
run_step="python src/MIL.py -c $CONFIG"

$run_step -s create_tiles
#$run_step -s extract_features_RetCCL
#$run_step -s create_splits
#$run_step -s train_model
#$run_step -s model_tune
#$run_step -s create_att_heatmaps
