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

# Fine-tune a pretrained multilingual encoder on the LAReQA retrieval task.

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-0}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}

TASK='lareqa'

# These settings should match those used in scripts/run_eval_lareqa.sh
MAX_SEQ_LEN=352  # Total sequence length (query + answer)
MAX_QUERY_LEN=96
MAX_ANSWER_LEN=256
NUM_EPOCHS=3.0
LR=2e-5

PER_GPU_BATCH_SIZE=4
GRAD_ACC_STEPS=4

if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert-retrieval"
  DO_LOWER_CASE=""
elif [ $MODEL == "xlm-roberta-large" ]; then
  MODEL_TYPE="xlmr-retrieval"
  DO_LOWER_CASE="--do_lower_case"
fi

MODEL_DIR=$OUT_DIR/$TASK/${MODEL}_LR${LR}_EPOCH${NUM_EPOCHS}_LEN${MAX_SEQ_LEN}
mkdir -p $MODEL_DIR

export CUDA_VISIBLE_DEVICES=$GPU

python third_party/run_retrieval_qa.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --train_file $DATA_DIR/squad/train-v1.1.json \
  --predict_file $DATA_DIR/squad/dev-v1.1.json \
  --per_gpu_train_batch_size $PER_GPU_BATCH_SIZE \
  --learning_rate $LR \
  --num_train_epochs $NUM_EPOCHS \
  --max_seq_length $MAX_SEQ_LEN \
  --max_query_length $MAX_QUERY_LEN \
  --max_answer_length $MAX_ANSWER_LEN \
  --logging_steps 1000 \
  --save_steps 1000 \
  --overwrite_output_dir \
  --gradient_accumulation_steps $GRAD_ACC_STEPS \
  --warmup_steps 0 \
  --output_dir $MODEL_DIR \
  --weight_decay 0.0 \
  --threads 8 \
  --train_lang en \
  --eval_lang en \
  $DO_LOWER_CASE
