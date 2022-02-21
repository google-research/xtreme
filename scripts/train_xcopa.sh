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

# Script to train a model on the English SIQa and the English COPA data and to
# produce predictions on the multilingual XCOPA test data.
# Note that running this script requires a newer version of Transformers
# (we used 4.9.2) as well as HuggingFace datasets (pip install datasets).

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-1}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=$GPU

LR=2e-5
EPOCH=5
MAXL=128
langs="et,ht,id,it,qu,sw,ta,th,tr,vi,zh"

LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  MAXL=128
  LR=3e-5
  BATCH_SIZE=2
  GRAD_ACC=16
else
  MAXL=128
  LR=2e-5
  BATCH_SIZE=8
  GRAD_ACC=4
fi

TASK=siqa

SIQA_SAVE_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}/"
mkdir -p $SIQA_SAVE_DIR

echo "Training on ${TASK}"

CUDA_VISIBLE_DEVICES=$GPU python third_party/run_xcopa.py \
  --task ${TASK} \
  --model_name_or_path $MODEL \
  --output_dir $SIQA_SAVE_DIR/ \
  --do_train \
  --do_eval \
  --do_predict \
  --overwrite_output_dir \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --max_seq_length $MAXL \
  --num_train_epochs $EPOCH

# Continue training on English COPA training set
TASK=xcopa

XCOPA_SAVE_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}/"
mkdir -p $XCOPA_SAVE_DIR

echo "Fine-tuning on ${TASK}"

# Note: XLM-R has trouble loading the tokenizer of an existing model
CUDA_VISIBLE_DEVICES=$GPU python third_party/run_xcopa.py \
  --task ${TASK} \
  --train_lang en \
  --tokenizer_name ${MODEL} \
  --predict_langs ${langs} \
  --model_name_or_path ${SIQA_SAVE_DIR} \
  --output_dir $XCOPA_SAVE_DIR/ \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --max_seq_length $MAXL \
  --num_train_epochs $EPOCH

# Note: We do this in a separate step so that if we just predict with the model,
# we still load the model trained on COPA.

echo "Predicting on ${TASK}"

CUDA_VISIBLE_DEVICES=$GPU python third_party/run_xcopa.py \
  --task ${TASK} \
  --train_lang en \
  --predict_langs ${langs} \
  --tokenizer_name ${MODEL} \
  --model_name_or_path ${XCOPA_SAVE_DIR} \
  --output_dir $XCOPA_SAVE_DIR/xcopa/ \
  --do_predict \
  --overwrite_output_dir
