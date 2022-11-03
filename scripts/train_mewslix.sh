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

# Fine-tune a pretrained multilingual encoder on the Mewsli-X retrieval task.
set -eu
REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-0}
DATA_DIR=${3:-"$REPO/download"}
OUT_DIR=${4:-"$REPO/outputs"}
TASK='mewslix'

# These settings should match those used in scripts/run_eval_mewslix.sh
# They are primarily aimed at a quick training time (~1h using 1 GPU for mBERT).
MAX_SEQ_LEN=64
NUM_EPOCHS=2
GRAD_ACC_STEPS=4

# Learning rates were set based on best dev-set loss on the English
# 'wikipedia_pairs-dev' after 2 epochs, searching over {1e-5, 2e-5, 5e-5, 1e-4}.
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert-retrieval"
  LR=2e-5
  DO_LOWER_CASE=""
  PER_GPU_BATCH_SIZE=64  # largest power of two that fit 16GB GPU RAM
  LOGGING_STEPS=50
  SAVE_STEPS=100
elif [ $MODEL == "xlm-roberta-large" ]; then
  MODEL_TYPE="xlmr-retrieval"
  LR=1e-5
  DO_LOWER_CASE="--do_lower_case"
  PER_GPU_BATCH_SIZE=8  # largest power of two that fit 16GB GPU RAM
  LOGGING_STEPS=500
  SAVE_STEPS=2000
else
  echo "$MODEL not configured."
fi

HYPER_INFO="LR${LR}_EPOCH${NUM_EPOCHS}_LEN${MAX_SEQ_LEN}_BS${PER_GPU_BATCH_SIZE}_ACC${GRAD_ACC_STEPS}"
MODEL_DIR="${OUT_DIR}/${TASK}/${MODEL}_${HYPER_INFO}"
mkdir -p $MODEL_DIR

export CUDA_VISIBLE_DEVICES=$GPU

echo $MODEL_DIR/train.log
python third_party/run_retrieval_el.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --data_dir $DATA_DIR/${TASK} \
  --train_file wikipedia_pairs-train.jsonl \
  --predict_file wikipedia_pairs-dev.jsonl \
  --per_gpu_train_batch_size $PER_GPU_BATCH_SIZE \
  --learning_rate $LR \
  --num_train_epochs $NUM_EPOCHS \
  --max_seq_length $MAX_SEQ_LEN \
  --logging_steps $LOGGING_STEPS \
  --save_steps $SAVE_STEPS \
  --overwrite_output_dir \
  --gradient_accumulation_steps $GRAD_ACC_STEPS \
  --warmup_steps 0 \
  --output_dir $MODEL_DIR \
  --weight_decay 0.0 \
  --threads 8 \
  --train_lang en \
  --eval_lang en \
  $DO_LOWER_CASE \
  2>&1 | tee $MODEL_DIR/train.log

set +x
bash $REPO/scripts/run_eval_mewslix.sh $MODEL_DIR $GPU $DATA_DIR $OUT_DIR
