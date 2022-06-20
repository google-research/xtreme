#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eu
REPO=$PWD
# Note: The default evaluates the frozen model. To evaluate a model fine-tuned
# with train_mewsli.x, pass its model directory.
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-0}
DATA_DIR=${3:-"$REPO/download"}
OUT_DIR=${4:-"$REPO/outputs"}
TASK='mewslix'

# These settings should match those used in scripts/train_mewslix.sh
MAX_SEQ_LEN=64
# Infer model type from the model argument, which could be a path.
MODEL_BASE="$(basename ${MODEL})"
# Note explicit lack of quotes below to get wildcard matching.
if [[ ${MODEL_BASE} == bert-base-multilingual-cased* ]]; then
  MODEL_TYPE="bert-retrieval"
  DIM=768
  DO_LOWER_CASE=""
elif [[ ${MODEL_BASE} == xlm-roberta-large* ]]; then
  MODEL_TYPE="xlmr-retrieval"
  DIM=1024
  DO_LOWER_CASE="--do_lower_case"
else
  echo "Failed to identify model type."
  exit
fi
if [[ -d "${MODEL}" ]]; then
  # When provided a directory, output the results to its subdirectory.
  OUTPUT_DIR="${MODEL}/run_eval"
else
  # Otherwise assume it is a vanilla pretrained (so no model directory) and just
  # output to a reasonable location:
  OUTPUT_DIR="$OUT_DIR/$TASK/${MODEL}_LEN${MAX_SEQ_LEN}/run_eval"
fi
mkdir -p $OUTPUT_DIR
export CUDA_VISIBLE_DEVICES=$GPU
python $REPO/third_party/evaluate_retrieval.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --embed_size $DIM \
  --batch_size 100 \
  --task_name $TASK \
  --pool_type cls \
  --max_seq_length $MAX_SEQ_LEN \
  --data_dir $DATA_DIR/$TASK \
  --output_dir $OUTPUT_DIR \
  --dist cosine \
  $DO_LOWER_CASE \
  2>&1 | tee $OUTPUT_DIR/eval.log