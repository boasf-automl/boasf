#!/usr/bin/env bash

TIME_BUDGET=7200
PER_MODEL_TIME_BUDGET=120
DATASET_BASE_PATH=$(dirname "$PWD")"/sample_data"
DATASET_IDX=37
GUCB_C=2.0
TEST_METRIC="balanced_accuracy_score"
MAX_ROUNDS=3

python ../test/openml_model_selection_boasf.py \
  --dataset_base_path=$DATASET_BASE_PATH \
  --dataset_idx=$DATASET_IDX \
  --time_budget=$TIME_BUDGET \
  --per_model_time_budget=$PER_MODEL_TIME_BUDGET \
  --gucb_c=$GUCB_C \
  --max_rounds=$MAX_ROUNDS \
  --test_metric=$TEST_METRIC
