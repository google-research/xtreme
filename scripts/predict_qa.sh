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

# Script to obtain predictions using a trained model on XQuAD, TyDi QA, and MLQA.
REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
MODEL_PATH=${2}
TGT=${3:-xquad}
GPU=${4:-0}
DATA_DIR=${5:-"$REPO/download/"}

if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlm-roberta"
fi

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model path ${MODEL_PATH} does not exist."
  exit
fi

DIR=${DATA_DIR}/${TGT}/
PREDICTIONS_DIR=${MODEL_PATH}/predictions
PRED_DIR=${PREDICTIONS_DIR}/$TGT/
mkdir -p "${PRED_DIR}"

if [ $TGT == 'xquad' ]; then
  langs=( en es de el ru tr ar vi th zh hi )
elif [ $TGT == 'mlqa' ]; then
  langs=( en es de ar hi vi zh )
elif [ $TGT == 'tydiqa' ]; then
  langs=( en ar bn fi id ko ru sw te )
fi

echo "************************"
echo ${MODEL}
echo "************************"

echo
echo "Predictions on $TGT"
for lang in ${langs[@]}; do
  echo "  $lang "
  if [ $TGT == 'xquad' ]; then
    TEST_FILE=${DIR}/xquad.$lang.json
  elif [ $TGT == 'mlqa' ]; then
    TEST_FILE=${DIR}/MLQA_V1/test/test-context-$lang-question-$lang.json
  elif [ $TGT == 'tydiqa' ]; then
    TEST_FILE=${DIR}/tydiqa-goldp-v1.1-dev/tydiqa.goldp.$lang.dev.json
  fi

  CUDA_VISIBLE_DEVICES=${GPU} python third_party/run_squad.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_PATH} \
    --do_eval \
    --eval_lang ${lang} \
    --predict_file "${TEST_FILE}" \
    --output_dir "${PRED_DIR}" &> /dev/null
done

# Rename files to test pattern
for lang in ${langs[@]}; do
  mv $PRED_DIR/predictions_${lang}_.json $PRED_DIR/test-$lang.json
done
