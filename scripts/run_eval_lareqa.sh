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

# Evaluate a fine-tuned model (trained using scripts/train_lareqa.sh) on the
# LAReQA retrieval task.

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-0}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}
# Select a checkpoint based on validation performance.
CHECKPOINT=${5:-checkpoint-9000}

TASK='lareqa'

# These settings should match those used in scripts/train_lareqa.sh
MAX_SEQ_LEN=352  # Total sequence length (query + answer)
MAX_QUERY_LEN=96
MAX_ANSWER_LEN=256
NUM_EPOCHS=3.0
LR=2e-5

if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert-retrieval"
  DIM=768
  DO_LOWER_CASE=""
elif [ $MODEL == "xlm-roberta-large" ]; then
  MODEL_TYPE="xlmr-retrieval"
  DIM=1024
  DO_LOWER_CASE="--do_lower_case"
fi

MODEL_DIR=$OUT_DIR/$TASK/${MODEL}_LR${LR}_EPOCH${NUM_EPOCHS}_LEN${MAX_SEQ_LEN}
MODEL_PATH=$MODEL_DIR/$CHECKPOINT
OUTPUT_DIR=$MODEL_DIR/eval_$CHECKPOINT
mkdir -p $OUTPUT_DIR

export CUDA_VISIBLE_DEVICES=$GPU

python $REPO/third_party/evaluate_retrieval.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_PATH \
  --embed_size $DIM \
  --batch_size 100 \
  --task_name $TASK \
  --pool_type cls \
  --max_seq_length $MAX_SEQ_LEN \
  --max_query_length $MAX_QUERY_LEN \
  --max_answer_length $MAX_ANSWER_LEN \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --extract_embeds \
  --dist cosine \
  $DO_LOWER_CASE
