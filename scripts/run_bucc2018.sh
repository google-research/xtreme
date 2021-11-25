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

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-0}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='bucc2018'
DATA_DIR=$DATA_DIR/$TASK/
MAXL=512
TL='en'

NLAYER=12
LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
  DIM=768
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  DIM=1280
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
  DIM=1024
  NLAYER=24
fi

SP='test'
for SL in fr ru zh de; do
  PRED_DIR=$REPO/predictions/
  OUT=$OUT_DIR/$TASK/$MODEL-${SL}
  mkdir -p $OUT
  for sp in 'test' 'dev'; do
    for lg in "$SL" "$TL"; do
      FILE=$DATA_DIR/${SL}-${TL}.${sp}.${lg}
      cut -f2 $FILE > $OUT/${SL}-${TL}.${sp}.${lg}.txt
      cut -f1 $FILE > $OUT/${SL}-${TL}.${sp}.${lg}.id
    done
  done

  CP="candidates"
  python $REPO/third_party/evaluate_retrieval.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --embed_size $DIM \
    --batch_size 100 \
    --task_name $TASK \
    --src_language $SL \
    --tgt_language $TL \
    --pool_type cls \
    --max_seq_length $MAXL \
    --data_dir $DATA_DIR \
    --output_dir $OUT \
    --predict_dir $PRED_DIR \
    --candidate_prefix $CP \
    --log_file mine-bitext-${SL}.log \
    --extract_embeds \
    --mine_bitext \
    --specific_layer 7 \
    --dist cosine $LC

done
