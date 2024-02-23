#!/bin/bash
export PL_DISABLE_FORK=1
export KMP_AFFINITY=disabled
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8

CONFIG=configs/ac_hero_tvt.yaml
run_step="python src/MIL.py -c $CONFIG"

#$run_step -s create_tiles
#$run_step -s reuse_9tissues
#$run_step -s extract_features3
#$run_step -s create_splits
#$run_step -s train_model
#$run_step -s train_many
#$run_step -s train_many_many
#$run_step -s inference_many_nonsense
#$run_step -s test_nonsense
#$run_step -s inference
$run_step -s inference_many_many
#$run_step -s model_tune
#$run_step -s create_att_heatmaps
